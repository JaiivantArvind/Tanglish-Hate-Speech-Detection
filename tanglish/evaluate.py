"""Metrics, dev-tuned decision offsets, significance tests and run summaries.

CLI:
    python -m tanglish.evaluate summarize --runs runs [--out reports/results.md]
    python -m tanglish.evaluate compare  --a runs/<exp_a> --b runs/<exp_b>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data import SHORT_LABELS

N = len(SHORT_LABELS)


def log_softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(N)), zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(N)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": {name: float(v) for name, v in zip(SHORT_LABELS, per_class)},
    }


def macro_f1(y_true, y_pred) -> float:
    """Same value as sklearn's macro F1 over all N labels, but fast enough for bootstrap loops."""
    cm = np.bincount(np.asarray(y_true) * N + np.asarray(y_pred), minlength=N * N).reshape(N, N)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(N), where=denom > 0)
    return float(f1.mean())


def tune_offsets(scores: np.ndarray, y: np.ndarray, grid=np.arange(-3.0, 3.01, 0.1), rounds: int = 3) -> np.ndarray:
    """Per-class additive offsets on log-probabilities that maximise dev macro F1.

    Coordinate ascent; class 0 stays fixed at 0 because only differences matter.
    Tune on dev, then apply the same offsets once to test.
    """
    logp = log_softmax(scores)
    offsets = np.zeros(scores.shape[1])
    best = macro_f1(y, logp.argmax(1))
    for _ in range(rounds):
        improved = False
        for k in range(1, scores.shape[1]):
            for v in grid:
                trial = offsets.copy()
                trial[k] = v
                f = macro_f1(y, (logp + trial).argmax(1))
                if f > best + 1e-9:
                    best, offsets, improved = f, trial, True
        if not improved:
            break
    return offsets


def paired_bootstrap(y, pred_a, pred_b, n: int = 10000, seed: int = 0) -> dict:
    """How often B beats A in macro F1 over test-set resamples."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        deltas[i] = macro_f1(y[idx], pred_b[idx]) - macro_f1(y[idx], pred_a[idx])
    return {
        "delta": macro_f1(y, pred_b) - macro_f1(y, pred_a),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_value": float((deltas <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# Run loading and summaries
# ---------------------------------------------------------------------------

def load_run(seed_dir: Path) -> dict:
    return {
        "metrics": json.loads((seed_dir / "metrics.json").read_text()),
        "dev_scores": np.load(seed_dir / "dev_scores.npy"),
        "test_scores": np.load(seed_dir / "test_scores.npy"),
        "dev_labels": np.load(seed_dir / "dev_labels.npy"),
        "test_labels": np.load(seed_dir / "test_labels.npy"),
    }


def completed_seeds(exp_dir: Path) -> list[Path]:
    return sorted(p for p in exp_dir.glob("seed*") if (p / "metrics.json").exists())


def seed_averaged(exp_dir: Path) -> dict:
    """Averages class probabilities over an experiment's seeds."""
    runs = [load_run(p) for p in completed_seeds(exp_dir)]
    if not runs:
        raise FileNotFoundError(f"no completed runs in {exp_dir}")
    avg = lambda key: np.log(np.mean([np.exp(log_softmax(r[key])) for r in runs], axis=0))
    return {"dev_scores": avg("dev_scores"), "test_scores": avg("test_scores"),
            "dev_labels": runs[0]["dev_labels"], "test_labels": runs[0]["test_labels"], "n_seeds": len(runs)}


def summarize(runs_dir: Path) -> list[dict]:
    rows = []
    for exp_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        seeds = completed_seeds(exp_dir)
        if not seeds:
            continue
        ms = [json.loads((s / "metrics.json").read_text()) for s in seeds]
        row = {"experiment": exp_dir.name, "seeds": len(ms)}
        for split in ("dev", "test", "test_tuned"):
            for key in ("macro_f1", "weighted_f1"):
                vals = [m[split][key] for m in ms]
                row[f"{split}_{key}"] = (float(np.mean(vals)), float(np.std(vals)))
        for name in SHORT_LABELS:
            row[f"test_f1_{name}"] = float(np.mean([m["test"]["per_class_f1"][name] for m in ms]))
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    fmt = lambda ms: f"{ms[0]:.4f} ± {ms[1]:.4f}"
    head = ("| experiment | seeds | dev macro-F1 | test macro-F1 | test weighted-F1 | test macro-F1 (dev-tuned offsets) | "
            + " | ".join(SHORT_LABELS) + " |")
    lines = [head, "|" + "---|" * (6 + len(SHORT_LABELS))]
    for r in rows:
        per = " | ".join(f"{r[f'test_f1_{n}']:.3f}" for n in SHORT_LABELS)
        lines.append(f"| {r['experiment']} | {r['seeds']} | {fmt(r['dev_macro_f1'])} | {fmt(r['test_macro_f1'])} | "
                     f"{fmt(r['test_weighted_f1'])} | {fmt(r['test_tuned_macro_f1'])} | {per} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--runs", default="runs")
    s.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="baseline experiment dir")
    c.add_argument("--b", required=True, help="proposed experiment dir")
    c.add_argument("--tuned", action="store_true", help="apply dev-tuned offsets before comparing")
    args = ap.parse_args()

    if args.cmd == "summarize":
        table = format_table(summarize(Path(args.runs)))
        print(table)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(table + "\n")
    else:
        a, b = seed_averaged(Path(args.a)), seed_averaged(Path(args.b))
        preds = []
        for run in (a, b):
            offsets = tune_offsets(run["dev_scores"], run["dev_labels"]) if args.tuned else 0.0
            preds.append((log_softmax(run["test_scores"]) + offsets).argmax(1))
        print(json.dumps(paired_bootstrap(a["test_labels"], preds[0], preds[1]), indent=2))


if __name__ == "__main__":
    main()
