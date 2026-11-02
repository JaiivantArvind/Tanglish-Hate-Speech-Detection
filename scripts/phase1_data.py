"""Phase 1: download the dataset and write a data audit (reports/data_stats.md).

    python scripts/phase1_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanglish.data import LABELS, load_splits, text_key  # noqa: E402
from tanglish.translit import script_of  # noqa: E402


def main(data_dir: str = "data", out: str = "reports/data_stats.md"):
    raw = load_splits(data_dir, clean=False)
    clean = load_splits(data_dir, clean=True)
    lines = ["# Dataset audit: DravidianCodeMix Tamil offensive (6 classes)", ""]

    lines += ["## Label counts", "", "| label | " + " | ".join(f"{s} raw" for s in raw) + " | train clean |",
              "|---|" + "---|" * (len(raw) + 1)]
    for i, name in enumerate(LABELS):
        counts = [int((raw[s]["label"] == i).sum()) for s in raw]
        lines.append(f"| {name} | " + " | ".join(map(str, counts)) + f" | {int((clean['train']['label'] == i).sum())} |")
    lines.append(f"| **total** | " + " | ".join(str(len(raw[s])) for s in raw) + f" | {len(clean['train'])} |")

    train_keys = raw["train"]["text"].map(text_key)
    test_keys = set(raw["test"]["text"].map(text_key))
    dev_keys = set(raw["dev"]["text"].map(text_key))
    lines += ["", "## Leakage and duplicates", "",
              f"- duplicate texts inside train: {int(train_keys.duplicated().sum())}",
              f"- test texts also in train: {sum(k in set(train_keys) for k in raw['test']['text'].map(text_key))}",
              f"- dev texts also in train: {sum(k in set(train_keys) for k in raw['dev']['text'].map(text_key))}",
              f"- train rows removed by cleaning: {len(raw['train']) - len(clean['train'])} "
              "(duplicates collapsed to the majority label, overlap with dev/test dropped)", ""]

    tr = raw["train"]
    scripts = tr["text"].map(script_of).value_counts(normalize=True)
    n_words = tr["text"].str.split().str.len()
    lines += ["## Text properties (train)", "",
              "- script share: " + ", ".join(f"{k} {v:.1%}" for k, v in scripts.items()),
              f"- words per comment: median {int(n_words.median())}, p95 {int(np.percentile(n_words, 95))}, "
              f"p99 {int(np.percentile(n_words, 99))}",
              f"- comments with @mentions: {int(tr['text'].str.contains('@').sum())}",
              f"- comments with URLs: {int(tr['text'].str.contains('http').sum())}", ""]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
