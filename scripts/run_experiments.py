"""Runs one stage of the experiment matrix; safe to re-launch after a session ends.

    python scripts/run_experiments.py --stage 1
    python scripts/run_experiments.py --stage 3 --dry-run

Stage 1: backbone sweep (sentence-only head, 1 seed)
Stage 2: domain-adaptive pretraining, then fine-tune the adapted encoder (1 seed).
        Select the backbones you care about, e.g.
        --only dapt_xlmr s2_xlmr_base_dapt
Stage 3: main ablation on the stage-3 backbone (3 seeds)
Stage 4: large backbones with the stage-3 winner (2 seeds)

Finished runs are skipped and interrupted runs resume, so the same command can
simply be launched again in a fresh Kaggle session.

A Kaggle session that hits the 12 h limit is cancelled, and a cancelled version
may not save its output at all, losing even the runs that had finished. So the
stage stops launching new runs once the next one would not fit in --max-hours,
exits 0, and lets the notebook finish and save normally.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS_MAIN = [13, 42, 87]

# Wall-clock minutes per run on 2x T4, used only to decide whether the next run
# fits in the remaining session budget. Measured for s3_cls; the rest are scaled
# from it. Replaced by this session's own timings as soon as a run of the same
# experiment finishes, so a wrong guess self-corrects.
DEFAULT_RUN_MINUTES = 45.0
RUN_MINUTES = {"s3_hier_tree_translit": 85.0, "s4_xlmr_large": 150.0, "s4_muril_large": 150.0}
DAPT_MINUTES = 120.0

STAGES = {
    1: [("s1_mbert", [42]), ("s1_xlmr_base", [42]), ("s1_muril_base", [42]), ("s1_indicbert2", [42])],
    2: [("s2_xlmr_base_dapt", [42]), ("s2_mbert_dapt", [42]), ("s2_muril_base_dapt", [42])],
    3: [(name, SEEDS_MAIN) for name in
        ("s3_cls", "s3_word", "s3_hier", "s3_hier_tree", "s3_hier_tree_lex", "s3_hier_tree_translit")],
    4: [("s4_xlmr_large", [13, 42])],  # s4_muril_large exists but MuRIL-base lost stage 1: use --only to add it
}

# Domain-adaptive pretraining jobs, selectable with --only.
DAPT_JOBS = {
    "dapt_xlmr": ("FacebookAI/xlm-roberta-base", "dapt/xlmr-base-tanglish"),
    "dapt_mbert": ("google-bert/bert-base-multilingual-cased", "dapt/mbert-tanglish"),
    "dapt_muril": ("google/muril-base-cased", "dapt/muril-base-tanglish"),
}


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


def run(cmd: list[str], dry: bool) -> float:
    """Launches one job and returns how many minutes it took."""
    print("$", shlex.join(cmd), flush=True)
    t0 = time.monotonic()
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)
    return (time.monotonic() - t0) / 60


def run_dir(name: str, seed: int) -> Path:
    return ROOT / "runs" / name / f"seed{seed}"


def remaining_fraction(name: str, seed: int) -> float:
    """How much of a run is left: 1.0 fresh, less when it resumes mid-training.

    A run that already has some epochs behind it costs proportionally less, so
    treating every run as full-price would stop the stage earlier than needed.
    """
    d = run_dir(name, seed)
    progress, config = d / "progress.json", d / "config.yaml"
    if not (progress.exists() and config.exists()):
        return 1.0
    try:
        done = json.loads(progress.read_text())["epoch"]
        # Avoid a YAML dependency here; the key is written one-per-line.
        epochs = next(int(line.split(":")[1]) for line in config.read_text().splitlines()
                      if line.startswith("epochs:"))
        return max(0.0, 1.0 - done / epochs) if epochs else 1.0
    except (ValueError, KeyError, StopIteration, json.JSONDecodeError):
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides passed to every run")
    ap.add_argument("--order", choices=["seed-major", "config-major"], default="seed-major",
                    help="seed-major finishes seed 1 of every config before starting seed 2, so an "
                         "interrupted stage still yields a complete ablation ranking")
    ap.add_argument("--max-hours", type=float, default=10.5,
                    help="stop launching runs once the next would not finish within this many hours; "
                         "0 disables. Keeps a Kaggle session below the 12 h cancel, which can discard "
                         "the whole version's output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    started = time.monotonic()
    budget = args.max_hours * 3600 if args.max_hours else None
    observed: dict[str, float] = {}
    launched = skipped = 0

    def fits(key: str, estimate: float) -> bool:
        """True if a job of this estimated length still fits in the session budget."""
        if budget is None:
            return True
        left = budget - (time.monotonic() - started)
        if estimate * 60 <= left:
            return True
        print(f"[budget] stopping before {key}: needs ~{estimate:.0f} min, "
              f"{left / 60:.0f} min left of the {args.max_hours:g} h budget.\n"
              f"[budget] re-launch this same command in a new session to continue.", flush=True)
        return False

    if args.stage == 2:
        for name, (model, out) in DAPT_JOBS.items():
            if args.only and name not in args.only:
                continue
            if (ROOT / out / "final").is_dir():
                print(f"[skip] {out} already built", flush=True)
                skipped += 1
                continue
            if not fits(name, DAPT_MINUTES):
                return 0
            run(launcher() + ["tanglish.dapt", "--model", model, "--out", out], args.dry_run)
            launched += 1

    jobs = [(name, seeds) for name, seeds in STAGES[args.stage]
            if not args.only or name in args.only]
    if args.order == "seed-major":
        # Round-robin over seed slots: every config's seed-13 run comes first, then
        # every config's seed-42 run, and so on. Configs may declare different seeds.
        pairs = [(name, seeds[i]) for i in range(max((len(s) for _, s in jobs), default=0))
                 for name, seeds in jobs if i < len(seeds)]
    else:
        pairs = [(name, seed) for name, seeds in jobs for seed in seeds]

    for name, seed in pairs:
        key = f"{name}/seed{seed}"
        # Checked here as well as in train.py, so a finished run costs no budget
        # and no interpreter start-up.
        if (run_dir(name, seed) / "metrics.json").exists():
            print(f"[skip] {key} already finished", flush=True)
            skipped += 1
            continue
        estimate = observed.get(name, RUN_MINUTES.get(name, DEFAULT_RUN_MINUTES)) * remaining_fraction(name, seed)
        if not fits(key, estimate):
            break
        cmd = launcher() + ["tanglish.train", "--config", f"configs/{name}.yaml", "--seed", str(seed)]
        if args.set:
            cmd += ["--set", *args.set]
        took = run(cmd, args.dry_run)
        launched += 1
        if not args.dry_run and took > 1:
            observed[name] = took

    print(f"[done] {launched} run(s) launched, {skipped} skipped, "
          f"{(time.monotonic() - started) / 3600:.1f} h elapsed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
