"""Tanglish offensive-term lexicon: curated semantic tags plus data-mined cues.

A matched word gets a tag token appended ("loosu" -> "loosu <LEX_STUPID>"). Tag
tokens are registered with the tokenizer as single added tokens, so the model
sees one learnable embedding per tag instead of subword fragments.

Mining uses the weighted log-odds ratio with an informative Dirichlet prior
(Monroe et al., 2008) between offensive and non-offensive comments, computed on
the training split only.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .preprocess import words

CURATED_PATH = Path(__file__).parent / "resources" / "lexicon.tsv"
MINED_TAG = "<LEX_CUE>"

# Label ids (see data.LABELS) that count as offensive for mining.
OFFENSIVE_IDS = {1, 2, 3, 4}
NOT_OFFENSIVE_ID = 0


def load_curated(path: Path = CURATED_PATH) -> dict[str, str]:
    lex = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            lex[row["term"].strip().lower()] = row["tag"].strip()
    return lex


def mine(train: pd.DataFrame, min_count: int = 5, prior_scale: float = 0.01) -> pd.DataFrame:
    """Scores every word by how strongly it marks offensive comments.

    Returns a DataFrame sorted by z-score with document counts per side.
    """
    off_docs = Counter()
    not_docs = Counter()
    for text, label in zip(train["text"], train["label"]):
        vocab = set(words(text))
        if label in OFFENSIVE_IDS:
            off_docs.update(vocab)
        elif label == NOT_OFFENSIVE_ID:
            not_docs.update(vocab)

    total = off_docs + not_docs
    n_total = sum(total.values())
    n_off = sum(off_docs.values())
    n_not = sum(not_docs.values())
    a0 = prior_scale * n_total

    rows = []
    for w, c in total.items():
        if c < min_count:
            continue
        a_w = a0 * c / n_total
        y_o, y_n = off_docs[w], not_docs[w]
        delta = math.log((y_o + a_w) / (n_off + a0 - y_o - a_w)) - math.log((y_n + a_w) / (n_not + a0 - y_n - a_w))
        var = 1.0 / (y_o + a_w) + 1.0 / (y_n + a_w)
        rows.append((w, delta / math.sqrt(var), y_o, y_n, y_o / c))
    df = pd.DataFrame(rows, columns=["term", "z", "offensive_docs", "not_offensive_docs", "offensive_rate"])
    return df.sort_values("z", ascending=False).reset_index(drop=True)


def mined_terms(train: pd.DataFrame, z_min: float = 3.0, rate_min: float = 0.75, min_count: int = 5) -> dict[str, str]:
    scores = mine(train, min_count=min_count)
    keep = scores[(scores["z"] >= z_min) & (scores["offensive_rate"] >= rate_min)]
    return {t: MINED_TAG for t in keep["term"]}


class Lexicon:
    def __init__(self, terms: dict[str, str]):
        self.terms = terms

    @classmethod
    def build(cls, mode: str, train: pd.DataFrame | None = None) -> "Lexicon | None":
        if mode == "off":
            return None
        terms: dict[str, str] = {}
        if mode in ("mined", "both"):
            if train is None:
                raise ValueError("mined lexicon needs the training split")
            terms.update(mined_terms(train))
        if mode in ("curated", "both"):
            terms.update(load_curated())  # curated tags win over the generic mined cue
        if mode not in ("mined", "curated", "both"):
            raise ValueError(f"unknown lexicon mode {mode!r}")
        return cls(terms)

    @property
    def tags(self) -> list[str]:
        return sorted(set(self.terms.values()))

    def apply(self, text: str) -> str:
        out = []
        for token in text.split(" "):
            out.append(token)
            tag = None
            for w in words(token):
                tag = self.terms.get(w)
                if tag:
                    break
            if tag:
                out.append(tag)
        return " ".join(out)

    def coverage(self, texts) -> float:
        texts = list(texts)
        hit = sum(any(w in self.terms for w in words(t)) for t in texts)
        return hit / max(1, len(texts))


TAG_PATTERN = re.compile(r"<LEX_[A-Z_]+>")
