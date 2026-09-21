"""Probability-averaging ensemble over finished experiments (CPU only).

    python -m tanglish.ensemble --runs runs --out reports/ensemble.json [--include s3_ s4_]

Each experiment contributes its seed-averaged probabilities. Members are chosen
by greedy forward selection with replacement on dev macro F1 (Caruana et al.,
2004); per-class offsets are then tuned on dev and applied once to test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import completed_seeds, log_softmax, macro_f1, metrics, seed_averaged, tune_offsets


def greedy_select(dev_probs: dict[str, np.ndarray], y: np.ndarray, max_members: int = 15) -> list[str]:
    chosen: list[str] = []
    total = np.zeros_like(next(iter(dev_probs.values())))
    best = -1.0
    for _ in range(max_members):
        scores = {name: macro_f1(y, (total + p).argmax(1)) for name, p in dev_probs.items()}
        name = max(scores, key=scores.get)
        if scores[name] <= best + 1e-9:
            break
        best = scores[name]
        chosen.append(name)
        total += dev_probs[name]
    return chosen


def build(runs_dir: str | Path, include: list[str] | None = None) -> tuple[dict, np.ndarray]:
    """Returns the ensemble result dict and its offset-tuned test predictions."""
    exps = [p for p in sorted(Path(runs_dir).iterdir()) if p.is_dir() and completed_seeds(p)
            and (not include or any(p.name.startswith(pre) for pre in include))]
    if not exps:
        raise FileNotFoundError(f"no finished experiments in {runs_dir}")
    runs = {p.name: seed_averaged(p) for p in exps}
    first = next(iter(runs.values()))
    dev_y, test_y = first["dev_labels"], first["test_labels"]
    to_probs = lambda s: np.exp(log_softmax(s))

    members = greedy_select({n: to_probs(r["dev_scores"]) for n, r in runs.items()}, dev_y)
    dev = np.log(np.mean([to_probs(runs[m]["dev_scores"]) for m in members], axis=0))
    test = np.log(np.mean([to_probs(runs[m]["test_scores"]) for m in members], axis=0))
    offsets = tune_offsets(dev, dev_y)
    tuned_pred = (log_softmax(test) + offsets).argmax(1)
    result = {
        "members": members,
        "dev": metrics(dev_y, dev.argmax(1)),
        "test": metrics(test_y, test.argmax(1)),
        "test_tuned": metrics(test_y, tuned_pred),
        "offsets": offsets.tolist(),
    }
    return result, tuned_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--include", nargs="*", default=[], help="experiment-name prefixes to consider")
    ap.add_argument("--out", default="reports/ensemble.json")
    args = ap.parse_args()

    result, _ = build(args.runs, args.include)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"members: {result['members']}")
    for k in ("dev", "test", "test_tuned"):
        print(f"{k:>10}: macro-F1 {result[k]['macro_f1']:.4f} | weighted-F1 {result[k]['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
