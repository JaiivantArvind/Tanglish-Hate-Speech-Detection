"""Runs one stage of the experiment matrix; safe to re-launch after a session ends.

    python scripts/run_experiments.py --stage 1
    python scripts/run_experiments.py --stage 3 --dry-run

Stage 1: backbone sweep (sentence-only head, 1 seed)
Stage 2: domain-adaptive pretraining of MuRIL and XLM-R, then fine-tune both (1 seed)
Stage 3: main ablation on the stage-3 backbone (3 seeds)
Stage 4: large backbones with the stage-3 winner (2 seeds)

Finished runs are skipped and interrupted runs resume, so the same command can
simply be launched again in a fresh Kaggle session.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS_MAIN = [13, 42, 87]

STAGES = {
    1: [("s1_mbert", [42]), ("s1_xlmr_base", [42]), ("s1_muril_base", [42]), ("s1_indicbert2", [42])],
    2: [("s2_muril_base_dapt", [42]), ("s2_xlmr_base_dapt", [42])],
    3: [(name, SEEDS_MAIN) for name in
        ("s3_cls", "s3_word", "s3_hier", "s3_hier_tree", "s3_hier_tree_lex", "s3_hier_tree_translit")],
    4: [("s4_muril_large", [13, 42]), ("s4_xlmr_large", [13, 42])],
}

DAPT_JOBS = [
    ("google/muril-base-cased", "dapt/muril-base-tanglish"),
    ("FacebookAI/xlm-roberta-base", "dapt/xlmr-base-tanglish"),
]


def launcher() -> list[str]:
    try:
        import torch
        n = torch.cuda.device_count()
    except ImportError:
        n = 0
    if n > 1:
        return [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu", f"--num_processes={n}",
                "--mixed_precision=fp16", "-m"]
    return [sys.executable, "-m"]


def run(cmd: list[str], dry: bool) -> None:
    print("$", shlex.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides passed to every run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.stage == 2:
        for model, out in DAPT_JOBS:
            run(launcher() + ["tanglish.dapt", "--model", model, "--out", out], args.dry_run)

    for name, seeds in STAGES[args.stage]:
        if args.only and name not in args.only:
            continue
        for seed in seeds:
            cmd = launcher() + ["tanglish.train", "--config", f"configs/{name}.yaml", "--seed", str(seed)]
            if args.set:
                cmd += ["--set", *args.set]
            run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
