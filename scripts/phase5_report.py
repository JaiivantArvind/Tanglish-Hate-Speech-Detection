"""Phase 5: results tables, significance tests, ensemble, figures and error analysis.

    python scripts/phase5_report.py [--runs runs] [--baseline s3_cls] [--proposed s3_hier_tree]

Everything is computed from saved logits, so this runs on CPU. The attention
plots additionally need the proposed model's best.pt (keep_checkpoint: true).
Writes into reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanglish import ensemble  # noqa: E402
from tanglish.config import load_config  # noqa: E402
from tanglish.data import LABELS, SHORT_LABELS, load_splits, subsample  # noqa: E402
from tanglish.evaluate import (completed_seeds, format_table, log_softmax, paired_bootstrap, seed_averaged,  # noqa: E402
                               summarize)
from tanglish.models import OffensiveClassifier  # noqa: E402
from tanglish.train import TextPipeline  # noqa: E402

SHARED_TASK = [
    ("Hate-Alert (EACL 2021 winner, transformer ensemble)", "0.78 weighted-F1"),
    ("Pseudo-labelling + transliteration, ULMFiT (Yasaswini et al. 2021)", "0.793 weighted-F1"),
]


def plot_confusion(y, pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y, pred, labels=list(range(len(LABELS))))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)), SHORT_LABELS)
    ax.set_yticks(range(len(LABELS)), SHORT_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def attention_plots(run_dir: Path, examples: list[str], out_dir: Path) -> list[str]:
    cfg = load_config(str(run_dir / "config.yaml"))
    if cfg.head not in ("word", "hier") or not (run_dir / "best.pt").exists():
        return []
    splits = load_splits(cfg.data_dir)
    pipe = TextPipeline(cfg, splits["train"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = OffensiveClassifier(cfg.model_name, cfg.head, cfg.label_tree, cfg.dropout, cfg.attn_dim)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()

    written = []
    for n, raw in enumerate(examples, 1):
        text = pipe.finish(pipe.base(raw))
        enc = tokenizer(text, truncation=True, max_length=cfg.max_len, return_tensors="pt")
        word_ids = [-1 if w is None else w for w in enc.word_ids(0)]
        with torch.no_grad():
            out = model(enc["input_ids"], enc["attention_mask"], torch.tensor([word_ids]))
        weights = out["attention"][0].numpy()
        pred = SHORT_LABELS[int(out["scores"].argmax())]
        n_words = max(word_ids) + 1
        spans = [enc.word_to_chars(0, w) for w in range(n_words)]
        labels = [text[s.start:s.end] if s else "?" for s in spans]

        fig, ax = plt.subplots(figsize=(max(6, 0.55 * n_words), 2.6))
        ax.bar(range(n_words), weights[:n_words], color="#4C72B0")
        ax.set_xticks(range(n_words), labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("word attention")
        ax.set_title(f"predicted {pred}: {raw[:70]}", fontsize=9)
        fig.tight_layout()
        path = out_dir / f"attention_{n}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path.name)
    return written


def error_analysis(test_df, y, base_pred, prop_pred, base_name, prop_name, k: int = 12) -> list[str]:
    fixed = np.where((base_pred != y) & (prop_pred == y))[0]
    broken = np.where((base_pred == y) & (prop_pred != y))[0]
    rng = np.random.default_rng(0)
    lines = [f"# Error analysis: {prop_name} vs {base_name} (test, seed-averaged)", "",
             f"- {base_name} wrong, {prop_name} right: {len(fixed)}",
             f"- {base_name} right, {prop_name} wrong: {len(broken)}", ""]
    for title, idx in (("Fixed by the proposed model", fixed), ("Broken by the proposed model", broken)):
        lines += [f"## {title}", "", "| text | true | baseline | proposed |", "|---|---|---|---|"]
        for i in rng.permutation(idx)[:k]:
            text = test_df["text"].iloc[i].replace("|", "/")[:160]
            lines.append(f"| {text} | {SHORT_LABELS[y[i]]} | {SHORT_LABELS[base_pred[i]]} | {SHORT_LABELS[prop_pred[i]]} |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--baseline", default="s3_cls")
    ap.add_argument("--proposed", default="s3_hier_tree")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. results table (mean ± std over seeds)
    rows = summarize(runs)
    if not rows:
        raise SystemExit(f"no finished runs in {runs}")
    md = ["# Results", "", "Test scores are mean ± std over seeds; macro-F1 is the primary metric.", "",
          format_table(rows), "", "Published results on the same 6-class test set:", ""]
    md += [f"- {name}: {score}" for name, score in SHARED_TASK]

    # 2. ensemble
    ens, ens_pred = ensemble.build(runs)
    (out / "ensemble.json").write_text(json.dumps(ens, indent=2))
    md += ["", "## Ensemble", "", f"- members: {', '.join(ens['members'])}",
           f"- test macro-F1 {ens['test']['macro_f1']:.4f}, weighted-F1 {ens['test']['weighted_f1']:.4f}",
           f"- with dev-tuned offsets: macro-F1 {ens['test_tuned']['macro_f1']:.4f}, "
           f"weighted-F1 {ens['test_tuned']['weighted_f1']:.4f}"]

    # 3. significance: every experiment vs the baseline (seed-averaged predictions)
    base_dir = runs / args.baseline
    if completed_seeds(base_dir):
        base = seed_averaged(base_dir)
        base_pred = base["test_scores"].argmax(1)
        md += ["", f"## Paired bootstrap vs {args.baseline} (test macro-F1, 10k resamples)", "",
               "| experiment | Δ macro-F1 | 95% CI | p (Δ ≤ 0) |", "|---|---|---|---|"]
        for r in rows:
            if r["experiment"] == args.baseline:
                continue
            other = seed_averaged(runs / r["experiment"])
            bs = paired_bootstrap(base["test_labels"], base_pred, other["test_scores"].argmax(1))
            md.append(f"| {r['experiment']} | {bs['delta']:+.4f} | [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}] | "
                      f"{bs['p_value']:.4f} |")

    # 4. confusion matrices
    best = max(rows, key=lambda r: r["dev_macro_f1"][0])["experiment"]
    best_run = seed_averaged(runs / best)
    plot_confusion(best_run["test_labels"], best_run["test_scores"].argmax(1),
                   f"{best} (best single experiment by dev macro-F1)", out / "confusion_best.png")
    plot_confusion(best_run["test_labels"], ens_pred, "Ensemble with dev-tuned offsets", out / "confusion_ensemble.png")
    md += ["", "## Figures", "", "- confusion_best.png, confusion_ensemble.png"]

    # 5. error analysis and attention plots for the proposed model
    prop_dir = runs / args.proposed
    test_df = load_splits(args.data_dir)["test"]
    limit = load_config(str(completed_seeds(runs / best)[0] / "config.yaml")).limit
    test_df = subsample(test_df, limit, 0)  # smoke runs evaluate on a fixed subsample
    if completed_seeds(prop_dir) and completed_seeds(base_dir):
        prop = seed_averaged(prop_dir)
        lines = error_analysis(test_df, prop["test_labels"], base["test_scores"].argmax(1),
                               prop["test_scores"].argmax(1), args.baseline, args.proposed)
        (out / "error_analysis.md").write_text("\n".join(lines) + "\n")
        md.append("- error_analysis.md")

        pred = prop["test_scores"].argmax(1)
        y = prop["test_labels"]
        conf = np.exp(log_softmax(prop["test_scores"])).max(1)
        roman = test_df["text"].map(lambda t: t.isascii()).to_numpy()
        picks = []
        for cls in (2, 3, 1, 0):  # individual, group, untargeted, not offensive
            idx = np.where((y == cls) & (pred == cls) & roman)[0]
            if len(idx):
                picks.append(test_df["text"].iloc[idx[np.argmax(conf[idx])]])
        for seed_dir in completed_seeds(prop_dir):
            written = attention_plots(seed_dir, picks + ["dei loosu poda get out da"], out)
            if written:
                md.append(f"- attention plots from {seed_dir}: {', '.join(written)}")
                break

    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
