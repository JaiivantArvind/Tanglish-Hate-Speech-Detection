"""Domain-adaptive pretraining: continued masked-LM training on in-domain Tanglish text.

    accelerate launch -m tanglish.dapt --model google/muril-base-cased --out dapt/muril-base-tanglish

Corpus: offensive train+dev plus sentiment train+dev comments, their script views,
and any --extra-text files (one comment per line). Every text that also occurs in
the offensive or sentiment *test* split is removed first (the sentiment splits
share about 3.5k comments with the offensive test set).

The result is loaded like any Hugging Face model: set `model_name: <out>/final`.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from .data import download_dataset, text_key
from .preprocess import normalize
from .translit import alt_view, load_cache


def read_texts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.split("\t")[0] for line in f if line.strip()]


def build_corpus(data_dir: str, extra_files: list[str], seed: int = 0) -> list[str]:
    folder = download_dataset(data_dir)
    banned = {text_key(t) for task in ("offensive", "sentiment")
              for t in read_texts(folder / f"tamil_{task}_full_test.csv")}

    raw = []
    for task in ("offensive", "sentiment"):
        for split in ("train", "dev"):
            raw += read_texts(folder / f"tamil_{task}_full_{split}.csv")
    for f in extra_files:
        raw += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]

    cache = load_cache(Path(data_dir) / "translit_cache.tsv")
    seen, corpus = set(), []
    for t in raw:
        k = text_key(t)
        if k in banned or k in seen:
            continue
        seen.add(k)
        base = normalize(t)
        corpus.append(base)
        alt = cache.get(base) or alt_view(base)
        if alt:
            corpus.append(alt)
    random.Random(seed).shuffle(corpus)
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--extra-text", nargs="*", default=[])
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    # The MLM head emits batch x length x vocab logits (250k vocab for XLM-R), which
    # Accelerate then upcasts to fp32. A small per-device batch keeps that tensor off
    # the 15 GB T4; grad_accum restores the effective batch.
    ap.add_argument("--batch-size", type=int, default=8, help="per device")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "final" / "config.json").exists():
        print(f"[skip] {out}/final already exists")
        return

    corpus = build_corpus(args.data_dir, args.extra_text, args.seed)
    print(f"DAPT corpus: {len(corpus)} texts")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = Dataset.from_dict({"text": corpus}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len),
        batched=True, remove_columns=["text"])
    split = ds.train_test_split(test_size=0.02, seed=args.seed)

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    # Integer warmup steps work on both transformers 4.x and 5.x (5.x dropped warmup_ratio).
    world = int(os.environ.get("WORLD_SIZE", "1"))
    total_steps = math.ceil(len(split["train"]) / (args.batch_size * world * args.grad_accum)) * args.epochs
    # Option names drift between transformers 4.x and 5.x; keep only what this version takes.
    wanted = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        group_by_length=True,          # batches of similar length: less padding, smaller logits
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,     # never accumulate vocab-sized eval logits
        warmup_steps=int(0.06 * total_steps),
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
    )
    supported = {f.name for f in dataclasses.fields(TrainingArguments)}
    dropped = sorted(set(wanted) - supported)
    if dropped:
        print(f"note: this transformers version ignores {dropped}")
    training_args = TrainingArguments(**{k: v for k, v in wanted.items() if k in supported})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm_probability=args.mlm_prob, pad_to_multiple_of=8),
    )
    last = max(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), default=None)
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    print(trainer.evaluate())
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    for ckpt in out.glob("checkpoint-*"):
        shutil.rmtree(ckpt)


if __name__ == "__main__":
    main()
