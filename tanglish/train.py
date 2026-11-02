"""Fine-tune one experiment for one seed.

    accelerate launch -m tanglish.train --config configs/s3_hier_tree.yaml --seed 13
    python -m tanglish.train --config configs/smoke.yaml            # single process

Writes runs/<name>/seed<seed>/: metrics.json, {dev,test}_{scores,labels}.npy,
config.yaml, tokenizer/, best.pt (optional). A finished run is skipped when
launched again, and an interrupted one resumes from its last completed epoch,
so a whole stage can be re-launched after a Kaggle session ends.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import ExpConfig, load_config, save_config
from .data import LABELS, load_splits, subsample
from .evaluate import log_softmax, metrics, tune_offsets
from .lexicon import Lexicon
from .losses import LossComputer
from .models import OffensiveClassifier
from .preprocess import normalize
from .translit import alt_view, load_cache

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class TextPipeline:
    """normalise -> optional script view -> optional lexicon tags."""

    def __init__(self, cfg: ExpConfig, train_df):
        self.lexicon = Lexicon.build(cfg.lexicon, train_df)
        self.translit_cache = load_cache(Path(cfg.data_dir) / "translit_cache.tsv") if cfg.translit != "off" else {}

    def base(self, text: str) -> str:
        return normalize(text)

    def alt(self, base_text: str) -> str | None:
        # Cached views include IndicXlit roman->Tamil output; otherwise fall back
        # to the rule-based Tamil->roman mapper.
        return self.translit_cache.get(base_text) or alt_view(base_text)

    def finish(self, text: str) -> str:
        return self.lexicon.apply(text) if self.lexicon else text


def encode(tokenizer, texts: list[str], labels: list[int], max_len: int) -> list[dict]:
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    items = []
    for i, label in enumerate(labels):
        word_ids = enc.word_ids(i)
        items.append({
            "input_ids": enc["input_ids"][i],
            "word_index": [-1 if w is None else w for w in word_ids],
            "label": label,
        })
    return items


class Collator:
    def __init__(self, pad_id: int, multiple: int = 8):
        self.pad_id, self.multiple = pad_id, multiple

    def __call__(self, batch: list[dict]) -> dict:
        length = max(len(b["input_ids"]) for b in batch)
        length = int(math.ceil(length / self.multiple) * self.multiple)
        ids = torch.full((len(batch), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.long)
        words = torch.full((len(batch), length), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = 1
            words[i, :n] = torch.tensor(b["word_index"])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "word_index": words, "labels": labels}


def build_split(pipe: TextPipeline, df, with_alt: bool) -> tuple[list[str], list[int], list[str | None]]:
    base = [pipe.base(t) for t in df["text"]]
    alts = [pipe.alt(t) for t in base] if with_alt else [None] * len(base)
    finished = [pipe.finish(t) for t in base]
    finished_alts = [pipe.finish(a) if a else None for a in alts]
    return finished, df["label"].tolist(), finished_alts


# ---------------------------------------------------------------------------
# Optimiser with layer-wise learning-rate decay
# ---------------------------------------------------------------------------

_LAYER = re.compile(r"\.layer\.(\d+)\.")
_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")


def param_groups(model: OffensiveClassifier, cfg: ExpConfig) -> list[dict]:
    n_layers = model.encoder.config.num_hidden_layers
    groups: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith("encoder."):
            lr = cfg.lr * cfg.head_lr_mult
        elif "embeddings" in name:
            lr = cfg.lr * cfg.llrd ** n_layers
        else:
            m = _LAYER.search(name)
            lr = cfg.lr * cfg.llrd ** (n_layers - 1 - int(m.group(1))) if m else cfg.lr
        wd = 0.0 if any(k in name for k in _NO_DECAY) else cfg.weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(model, loader, accelerator) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        out = model(batch["input_ids"], batch["attention_mask"], batch["word_index"])
        s, y = accelerator.gather_for_metrics((out["scores"].float().contiguous(), batch["labels"]))
        scores.append(s.cpu())
        labels.append(y.cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def predict_with_views(model, accelerator, tokenizer, collate, cfg, texts, labels, alts, use_tta: bool):
    """Scores for a split; with TTA, averages probabilities over both script views."""
    loader = accelerator.prepare(DataLoader(encode(tokenizer, texts, labels, cfg.max_len),
                                            batch_size=cfg.batch_size * 2, collate_fn=collate))
    scores, y = predict(model, loader, accelerator)
    idx = [i for i, a in enumerate(alts) if a]
    if not use_tta or not idx:
        return scores, y
    alt_loader = accelerator.prepare(DataLoader(encode(tokenizer, [alts[i] for i in idx], [labels[i] for i in idx], cfg.max_len),
                                                batch_size=cfg.batch_size * 2, collate_fn=collate))
    alt_scores, _ = predict(model, alt_loader, accelerator)
    probs = np.exp(log_softmax(scores))
    probs[idx] = 0.5 * (probs[idx] + np.exp(log_softmax(alt_scores)))
    return np.log(probs), y


def run(cfg: ExpConfig) -> dict | None:
    run_dir = cfg.run_dir
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir} already finished")
        return json.loads((run_dir / "metrics.json").read_text())

    use_cuda = torch.cuda.is_available()
    accelerator = Accelerator(mixed_precision="fp16" if (cfg.fp16 and use_cuda) else "no",
                              gradient_accumulation_steps=cfg.grad_accum)
    set_seed(cfg.seed)
    log = accelerator.print
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, run_dir / "config.yaml")

    # ---- data ----
    splits = load_splits(cfg.data_dir, verbose=accelerator.is_main_process)
    if cfg.limit:
        # Dev/test subsamples must not depend on the seed, or seeds could not be averaged.
        splits = {k: subsample(v, cfg.limit, cfg.seed if k == "train" else 0) for k, v in splits.items()}
    pipe = TextPipeline(cfg, splits["train"])
    augment = cfg.translit in ("aug", "aug_tta")
    tta = cfg.translit in ("tta", "aug_tta")

    train_texts, train_labels, train_alts = build_split(pipe, splits["train"], with_alt=augment)
    if augment:
        extra = [(a, y) for a, y in zip(train_alts, train_labels) if a]
        train_texts += [a for a, _ in extra]
        train_labels += [y for _, y in extra]
        log(f"translit augmentation: +{len(extra)} training views")
    dev = build_split(pipe, splits["dev"], with_alt=tta)
    test = build_split(pipe, splits["test"], with_alt=tta)
    if pipe.lexicon:
        log(f"lexicon '{cfg.lexicon}': {len(pipe.lexicon.terms)} terms, train coverage "
            f"{pipe.lexicon.coverage(splits['train']['text']):.1%}")

    # A Hub id looks like "namespace/name"; anything deeper is a local directory
    # (e.g. a DAPT output), which must exist before training can start.
    if cfg.model_name.count("/") > 1 and not Path(cfg.model_name).is_dir():
        raise FileNotFoundError(
            f"model_name '{cfg.model_name}' is a local path but does not exist. "
            "On Kaggle: attach the previous version (Add Input -> Your Work) and run the "
            "restore cell so dapt/ is copied back, or point the config at a Hub model, "
            "e.g. --set model_name=FacebookAI/xlm-roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if not tokenizer.is_fast:
        raise RuntimeError("word pooling needs a fast tokenizer (word_ids)")
    if pipe.lexicon:
        tokenizer.add_tokens(pipe.lexicon.tags)
    collate = Collator(tokenizer.pad_token_id)

    train_loader = DataLoader(encode(tokenizer, train_texts, train_labels, cfg.max_len), batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=cfg.num_workers, drop_last=False)
    dev_loader = DataLoader(encode(tokenizer, dev[0], dev[1], cfg.max_len), batch_size=cfg.batch_size * 2,
                            collate_fn=collate, num_workers=cfg.num_workers)

    # ---- model ----
    model = OffensiveClassifier(cfg.model_name, cfg.head, cfg.label_tree, cfg.dropout, cfg.attn_dim)
    if pipe.lexicon:
        model.encoder.resize_token_embeddings(len(tokenizer))
    loss_fn = LossComputer(train_labels, cfg.label_tree, cfg.loss, cfg.class_weight,
                           cfg.focal_gamma, cfg.label_smoothing).to(accelerator.device)

    optimizer = torch.optim.AdamW(param_groups(model, cfg))
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler)

    # ---- resume ----
    state_dir, progress_file, best_file = run_dir / "state", run_dir / "progress.json", run_dir / "best.pt"
    progress = {"epoch": 0, "best_dev_macro_f1": -1.0, "bad_epochs": 0, "history": []}
    if progress_file.exists() and state_dir.exists():
        progress = json.loads(progress_file.read_text())
        accelerator.load_state(str(state_dir))
        log(f"resumed after epoch {progress['epoch']}")

    # ---- train ----
    for epoch in range(progress["epoch"] + 1, cfg.epochs + 1):
        if progress["bad_epochs"] >= cfg.patience:
            break
        set_seed(cfg.seed + epoch)
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                out = model(batch["input_ids"], batch["attention_mask"], batch["word_index"])
                loss = loss_fn(out, batch["labels"])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.item()
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        dev_scores, dev_y = predict(model, dev_loader, accelerator)
        dev_m = metrics(dev_y, dev_scores.argmax(1))
        improved = dev_m["macro_f1"] > progress["best_dev_macro_f1"]
        progress["history"].append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)),
                                    "dev_macro_f1": dev_m["macro_f1"], "dev_weighted_f1": dev_m["weighted_f1"],
                                    "minutes": (time.time() - t0) / 60})
        log(f"epoch {epoch}: train loss {progress['history'][-1]['train_loss']:.4f} | dev macro-F1 "
            f"{dev_m['macro_f1']:.4f} | dev weighted-F1 {dev_m['weighted_f1']:.4f} | "
            f"{progress['history'][-1]['minutes']:.1f} min{' *' if improved else ''}")
        if improved:
            progress["best_dev_macro_f1"], progress["bad_epochs"] = dev_m["macro_f1"], 0
            accelerator.wait_for_everyone()
            accelerator.save(accelerator.unwrap_model(model).state_dict(), best_file)
        else:
            progress["bad_epochs"] += 1
        progress["epoch"] = epoch
        accelerator.wait_for_everyone()
        accelerator.save_state(str(state_dir))
        if accelerator.is_main_process:
            progress_file.write_text(json.dumps(progress, indent=2))

    # ---- final evaluation with the best checkpoint ----
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).load_state_dict(torch.load(best_file, map_location="cpu"))
    dev_scores, dev_y = predict_with_views(model, accelerator, tokenizer, collate, cfg, *dev, use_tta=tta)
    test_scores, test_y = predict_with_views(model, accelerator, tokenizer, collate, cfg, *test, use_tta=tta)

    result = None
    if accelerator.is_main_process:
        offsets = tune_offsets(dev_scores, dev_y)
        result = {
            "config": dataclasses.asdict(cfg),
            "labels": LABELS,
            "dev": metrics(dev_y, dev_scores.argmax(1)),
            "test": metrics(test_y, test_scores.argmax(1)),
            "dev_tuned": metrics(dev_y, (log_softmax(dev_scores) + offsets).argmax(1)),
            "test_tuned": metrics(test_y, (log_softmax(test_scores) + offsets).argmax(1)),
            "offsets": offsets.tolist(),
            "history": progress["history"],
            "train_size": len(train_labels),
        }
        for name, arr in [("dev_scores", dev_scores), ("dev_labels", dev_y),
                          ("test_scores", test_scores), ("test_labels", test_y)]:
            np.save(run_dir / f"{name}.npy", arr)
        tokenizer.save_pretrained(run_dir / "tokenizer")
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        shutil.rmtree(state_dir, ignore_errors=True)
        progress_file.unlink(missing_ok=True)
        if not cfg.keep_checkpoint:
            best_file.unlink(missing_ok=True)
        log(f"TEST macro-F1 {result['test']['macro_f1']:.4f} | weighted-F1 {result['test']['weighted_f1']:.4f} | "
            f"macro-F1 with dev-tuned offsets {result['test_tuned']['macro_f1']:.4f}")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, key=value")
    args = ap.parse_args()
    overrides = list(args.set) + ([f"seed={args.seed}"] if args.seed is not None else [])
    run(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
