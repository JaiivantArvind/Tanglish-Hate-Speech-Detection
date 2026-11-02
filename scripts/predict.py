"""Classify Tanglish comments with a trained model.

    python scripts/predict.py --run demo/s3b_word_lex/seed13 --file comments.txt
    python scripts/predict.py --run demo/s3b_word_lex/seed13 --text "dei loosu poda"
    python scripts/predict.py --run demo/s3b_word_lex/seed13            # type comments, one per line

Needs a run that kept its weights (`keep_checkpoint: true`). Most runs do not: see
the note in tanglish/config.py. Re-run one with
`--set out_dir=demo keep_checkpoint=true` to get a copy with its weights.

--file reads one comment per line, ignoring blanks and lines starting with '#'.
--tuned applies the per-class offsets chosen on the practice set, the same ones
used for the reported "dev-tuned" scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanglish.config import load_config  # noqa: E402
from tanglish.data import LABELS, SHORT_LABELS, load_splits  # noqa: E402
from tanglish.lexicon import OFFENSIVE_IDS  # noqa: E402
from tanglish.models import OffensiveClassifier  # noqa: E402
from tanglish.train import TextPipeline  # noqa: E402

# The same four classes the rest of the codebase counts as offensive, so the
# binary verdict cannot drift from the label definitions in tanglish/data.py.
OFFENSIVE_LABELS = {LABELS[i] for i in OFFENSIVE_IDS}

READABLE = {
    "Not_offensive": "not offensive",
    "Offensive_Untargetede": "offensive, aimed at no one",
    "Offensive_Targeted_Insult_Individual": "offensive, aimed at a person",
    "Offensive_Targeted_Insult_Group": "offensive, aimed at a group",
    "Offensive_Targeted_Insult_Other": "offensive, aimed at something else",
    "not-Tamil": "not Tamil",
}


# The DAPT encoder is a plain XLM-R base with different weights, so when the local
# DAPT directory is absent the Hub model supplies an identical architecture.
FALLBACK_ENCODER = "FacebookAI/xlm-roberta-base"


class Predictor:
    def __init__(self, run_dir: Path, tuned: bool = False, encoder: str | None = None):
        from transformers import AutoTokenizer

        self.cfg = load_config(str(run_dir / "config.yaml"))
        weights = run_dir / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found: this run did not keep its weights. "
                "Re-run it with --set out_dir=demo keep_checkpoint=true.")

        # Only the architecture is needed here: best.pt supplies every weight below.
        encoder_name = encoder or self.cfg.model_name
        if encoder_name.count("/") > 1 and not Path(encoder_name).is_dir():
            print(f"note: '{encoder_name}' is not present, using {FALLBACK_ENCODER} "
                  "for the architecture (the checkpoint supplies the weights)")
            encoder_name = FALLBACK_ENCODER

        train_df = load_splits(self.cfg.data_dir)["train"] if self.cfg.lexicon in ("mined", "both") else None
        self.pipe = TextPipeline(self.cfg, train_df)
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        self.model = OffensiveClassifier(encoder_name, self.cfg.head, self.cfg.label_tree,
                                         self.cfg.dropout, self.cfg.attn_dim, pretrained=False)
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

        self.offsets = torch.zeros(len(LABELS))
        if tuned:
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.offsets = torch.tensor(metrics["offsets"], dtype=torch.float)

    @torch.no_grad()
    def __call__(self, text: str) -> tuple[list[tuple[str, float]], str, list[tuple[str, float]]]:
        processed = self.pipe.finish(self.pipe.base(text))
        enc = self.tokenizer(processed, truncation=True, max_length=self.cfg.max_len, return_tensors="pt")
        word_index = torch.tensor([[-1 if w is None else w for w in enc.word_ids(0)]])
        out = self.model(enc["input_ids"], enc["attention_mask"], word_index)
        probs = torch.softmax(out["scores"][0] + self.offsets, dim=-1)
        ranked = sorted(zip(LABELS, probs.tolist()), key=lambda kv: -kv[1])

        attention = []
        if out["attention"] is not None:
            weights = out["attention"][0]
            n_words = int(word_index.max()) + 1
            spans = [enc.word_to_chars(0, w) for w in range(n_words)]
            attention = sorted(((processed[s.start:s.end], float(weights[w])) for w, s in enumerate(spans) if s),
                               key=lambda kv: -kv[1])[:5]
        return ranked, processed, attention


def offensive_prob(ranked: list[tuple[str, float]]) -> float:
    """P(offensive) = the four Offensive_* classes summed.

    More reliable than the six-way argmax: the model separates offensive from not
    far better than it tells the four offensive types apart.
    """
    return sum(p for name, p in ranked if name in OFFENSIVE_LABELS)


def report(pred: Predictor, text: str) -> None:
    ranked, processed, attention = pred(text)
    label, prob = ranked[0]
    off = offensive_prob(ranked)
    print(f"\n  \"{text}\"")
    print(f"  OFFENSIVE: {'yes' if off >= 0.5 else 'no':<3}  ({off:.0%} offensive, {1 - off:.0%} not)")
    print(f"  type:      {READABLE[label]} ({prob:.0%} confident)")
    print(f"  text the model saw: {processed}")
    print("  all six:", ", ".join(f"{READABLE[n]} {p:.0%}" for n, p in ranked))
    if attention:
        print("  words it focused on:", ", ".join(f"{w} {a:.0%}" for w, a in attention))


def read_comments(path: Path) -> list[tuple[str, int | None]]:
    """One comment per line; blank lines and '#' comments are ignored.

    A line may optionally carry a gold label after a tab ("comment\\tNOT"), in
    which case the run scores itself. Plain .txt files work unchanged.
    """
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        text, _, gold = line.partition("\t")
        text, gold = text.strip(), gold.strip()
        label = None
        if gold:
            names = {n.upper(): i for i, n in enumerate(SHORT_LABELS)}
            names.update({n.upper(): i for i, n in enumerate(LABELS)})
            if gold.upper() not in names:
                raise SystemExit(f"unknown label {gold!r} in {path}; expected one of {SHORT_LABELS}")
            label = names[gold.upper()]
        out.append((text, label))
    return out


def print_accuracy(rows: list[tuple[str, str, float, float]], gold: list[int | None]) -> None:
    """Scored only over the lines that carried a gold label."""
    pairs = [(r, g) for r, g in zip(rows, gold) if g is not None]
    if not pairs:
        return
    n = len(pairs)
    six = sum(1 for (_, label, _, _), g in pairs if LABELS.index(label) == g)
    majority = sum(1 for _, g in pairs if g == 0)

    # Binary confusion: "offensive" is the positive class, so a false positive is
    # a clean comment wrongly flagged and a false negative is one that slipped by.
    tp = sum(1 for (_, _, _, off), g in pairs if off >= 0.5 and g in OFFENSIVE_IDS)
    fp = sum(1 for (_, _, _, off), g in pairs if off >= 0.5 and g not in OFFENSIVE_IDS)
    fn = sum(1 for (_, _, _, off), g in pairs if off < 0.5 and g in OFFENSIVE_IDS)
    tn = sum(1 for (_, _, _, off), g in pairs if off < 0.5 and g not in OFFENSIVE_IDS)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"\nscored against gold labels on {n} comments")
    print(f"\n  offensive / not                 flagged   not flagged")
    print(f"    truly offensive ({tp + fn:>3})     TP {tp:>4}      FN {fn:>4}")
    print(f"    truly not       ({fp + tn:>3})     FP {fp:>4}      TN {tn:>4}")
    print(f"\n    accuracy  {(tp + tn) / n:>6.1%}   ({tp + tn} of {n} correct)")
    print(f"    precision {precision:>6.1%}   (of {tp + fp} flagged, {tp} were really offensive)")
    print(f"    recall    {recall:>6.1%}   (of {tp + fn} offensive, {tp} were caught)")
    print(f"    F1        {f1:>6.1%}")
    print(f"\n  exact type (6-way)  {six / n:.1%}"
          f"   (always saying 'not offensive' would give {majority / n:.1%})")

    print("\n  per class:")
    for i, short in enumerate(SHORT_LABELS):
        total = sum(1 for _, g in pairs if g == i)
        if not total:
            continue
        hit = sum(1 for (_, label, _, _), g in pairs if g == i and LABELS.index(label) == i)
        print(f"    {short}  n={total:>3}  found {hit:>3}  ({hit / total:>5.1%})")

    wrong = [(i, r[0], r[1], g) for i, (r, g) in enumerate(zip(rows, gold), 1)
             if g is not None and LABELS.index(r[1]) != g]
    if wrong:
        print(f"\n  {len(wrong)} wrong on the six-way label:")
        for i, text, label, g in wrong[:12]:
            print(f"    {i:>3}  said {SHORT_LABELS[LABELS.index(label)]:<4} was {SHORT_LABELS[g]:<4}  {text[:52]}")
        if len(wrong) > 12:
            print(f"    ... and {len(wrong) - 12} more")


def classify_all(pred: Predictor, comments: list[str]) -> list[tuple[str, str, float, float]]:
    """Each row: text, six-way label, that label's probability, P(offensive)."""
    rows = []
    for text in comments:
        ranked, _, _ = pred(text)
        label, prob = ranked[0]
        rows.append((text, label, prob, offensive_prob(ranked)))
    return rows


def print_table(rows: list[tuple[str, str, float, float]], width: int = 50) -> None:
    # The comment goes last: Tamil script is not as wide as len() suggests, so
    # padding it would throw off every column after it.
    n_w = max(2, len(str(len(rows))))
    label_w = max(len(READABLE[l]) for _, l, _, _ in rows)
    print(f"\n{'#':>{n_w}}  {'offensive?':<10}  {'type':<{label_w}}  comment")
    print(f"{'-' * n_w}  {'-' * 10}  {'-' * label_w}  {'-' * width}")
    for i, (text, label, _, off) in enumerate(rows, 1):
        verdict = f"{'YES' if off >= 0.5 else 'no':<3} {off:>3.0%}"
        shown = text if len(text) <= width else text[: width - 1] + "…"
        print(f"{i:>{n_w}}  {verdict:<10}  {READABLE[label]:<{label_w}}  {shown}")


def print_summary(rows: list[tuple[str, str, float, float]]) -> None:
    offensive = sum(1 for _, _, _, off in rows if off >= 0.5)
    print(f"\n{len(rows)} comments: {offensive} offensive, {len(rows) - offensive} not")

    counts = Counter(label for _, label, _, _ in rows)
    print("\nbreakdown by type (the six-way label):")
    for label in LABELS:
        if counts[label]:
            print(f"  {counts[label]:>3}  {READABLE[label]}")

    # Anything near the 50% line is effectively a coin flip and worth eyeballing.
    unsure = [(i, t, off) for i, (t, _, _, off) in enumerate(rows, 1) if 0.35 <= off <= 0.65]
    if unsure:
        print("\nthe model is unsure about these (offensive 35-65%):")
        for i, text, off in unsure:
            print(f"  {i:>3}  {off:>3.0%}  {text[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run directory, e.g. demo/s3b_word_lex/seed13")
    ap.add_argument("--file", type=Path, help="a .txt of comments, one per line")
    ap.add_argument("--text", nargs="*", help="comments to classify; omit to type them one per line")
    ap.add_argument("--tuned", action="store_true", help="apply the practice-set offsets")
    ap.add_argument("--detail", action="store_true",
                    help="with --file: also show all six probabilities and the attended words")
    ap.add_argument("--out", type=Path, help="with --file: write text,label,confidence as CSV")
    ap.add_argument("--encoder", help="override the encoder path/id the run was built on")
    args = ap.parse_args()

    pred = Predictor(Path(args.run), args.tuned, args.encoder)
    print(f"loaded {args.run} ({pred.cfg.model_name}, head={pred.cfg.head}, "
          f"lexicon={pred.cfg.lexicon}, three-question heads={'on' if pred.cfg.label_tree else 'off'})")

    if args.file:
        entries = read_comments(args.file)
        if not entries:
            raise SystemExit(f"no comments in {args.file}")
        comments = [t for t, _ in entries]
        gold = [g for _, g in entries]
        rows = classify_all(pred, comments)
        print_table(rows)
        print_summary(rows)
        print_accuracy(rows, gold)
        if args.detail:
            for text, _, _, _ in rows:
                report(pred, text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["text", "offensive", "offensive_prob", "type", "type_prob"])
                w.writerows((t, "yes" if off >= 0.5 else "no", f"{off:.4f}", READABLE[l], f"{p:.4f}")
                            for t, l, p, off in rows)
            print(f"\nwrote {args.out}")
        return

    if args.text:
        for text in args.text:
            report(pred, text)
        return
    print("Type a comment and press enter. Ctrl-C to stop.")
    try:
        for line in sys.stdin:
            if line.strip():
                report(pred, line.strip())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
