"""Classify typed comments with a trained model.

    python scripts/predict.py --run runs/s3_hier_tree/seed13 --text "dei loosu poda"
    python scripts/predict.py --run runs/s3_hier_tree/seed13            # type comments, one per line

Needs a run that kept its weights (`keep_checkpoint: true`, set on s3_hier_tree).
--tuned applies the per-class offsets chosen on the practice set, the same ones
used for the reported "dev-tuned" scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanglish.config import load_config  # noqa: E402
from tanglish.data import LABELS, load_splits  # noqa: E402
from tanglish.models import OffensiveClassifier  # noqa: E402
from tanglish.train import TextPipeline  # noqa: E402

READABLE = {
    "Not_offensive": "not offensive",
    "Offensive_Untargetede": "offensive, aimed at no one",
    "Offensive_Targeted_Insult_Individual": "offensive, aimed at a person",
    "Offensive_Targeted_Insult_Group": "offensive, aimed at a group",
    "Offensive_Targeted_Insult_Other": "offensive, aimed at something else",
    "not-Tamil": "not Tamil",
}


class Predictor:
    def __init__(self, run_dir: Path, tuned: bool = False):
        from transformers import AutoTokenizer

        self.cfg = load_config(str(run_dir / "config.yaml"))
        weights = run_dir / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"{weights} not found: this run did not keep its weights. "
                "Re-run it with --set keep_checkpoint=true, or use a run that did (s3_hier_tree).")

        train_df = load_splits(self.cfg.data_dir)["train"] if self.cfg.lexicon in ("mined", "both") else None
        self.pipe = TextPipeline(self.cfg, train_df)
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        self.model = OffensiveClassifier(self.cfg.model_name, self.cfg.head, self.cfg.label_tree,
                                         self.cfg.dropout, self.cfg.attn_dim)
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


def report(pred: Predictor, text: str) -> None:
    ranked, processed, attention = pred(text)
    label, prob = ranked[0]
    print(f"\n  {READABLE[label]}  ({prob:.0%} confident)")
    print(f"  text the model saw: {processed}")
    print("  all six:", ", ".join(f"{READABLE[n]} {p:.0%}" for n, p in ranked))
    if attention:
        print("  words it focused on:", ", ".join(f"{w} {a:.0%}" for w, a in attention))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run directory, e.g. runs/s3_hier_tree/seed13")
    ap.add_argument("--text", nargs="*", help="comments to classify; omit to type them one per line")
    ap.add_argument("--tuned", action="store_true", help="apply the practice-set offsets")
    args = ap.parse_args()

    pred = Predictor(Path(args.run), args.tuned)
    print(f"loaded {args.run} ({pred.cfg.model_name}, head={pred.cfg.head}, "
          f"three-question heads={'on' if pred.cfg.label_tree else 'off'})")

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
