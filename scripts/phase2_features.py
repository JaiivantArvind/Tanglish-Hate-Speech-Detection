"""Phase 2: lexicon analysis and the transliteration cache.

    python scripts/phase2_features.py            # lexicon report + rule-based views
    python scripts/phase2_features.py --xlit     # also romanised -> Tamil with IndicXlit

Writes
  reports/lexicon_report.md       how the guide's lexicon and the curated lexicon behave
  reports/lexicon_candidates.tsv  top mined terms, for extending tanglish/resources/lexicon.tsv
  data/translit_cache.tsv         normalised text -> other-script view
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tanglish.data import load_splits  # noqa: E402
from tanglish.lexicon import Lexicon, load_curated, mine  # noqa: E402
from tanglish.preprocess import normalize, words  # noqa: E402
from tanglish.translit import XlitRomanToTamil, alt_view, load_cache, save_cache, script_of  # noqa: E402

# The lexicon from the original guide, kept to show why it was replaced.
GUIDE_TERMS = ["poda", "podi", "ponga", "tholu", "loosu", "losu", "kena", "vaangai", "mutti", "naaye", "naaya",
               "naye", "kazhudhai", "panni", "thevdiya", "thevidiya", "otha", "oombu", "baadu", "paarpan",
               "parpan", "dei", "daa"]


def term_table(train, terms) -> list[str]:
    rows = ["| term | comments | % not offensive |", "|---|---|---|"]
    for t in terms:
        hits = train[train["text"].map(lambda s: t in words(s))]
        if len(hits):
            rows.append(f"| {t} | {len(hits)} | {100 * (hits['label'] == 0).mean():.0f}% |")
        else:
            rows.append(f"| {t} | 0 | - |")
    return rows


def lexicon_report(splits, out_dir: Path) -> None:
    train = splits["train"]
    guide = Lexicon({t: "<TAG>" for t in GUIDE_TERMS})
    curated = Lexicon(load_curated())
    lines = ["# Lexicon analysis (train split)", "",
             f"- guide lexicon: {len(GUIDE_TERMS)} terms, covers {guide.coverage(train['text']):.1%} of comments",
             f"- curated lexicon: {len(curated.terms)} terms, covers {curated.coverage(train['text']):.1%} of comments",
             "", "## Guide lexicon: how often each term appears in non-offensive comments", ""]
    lines += term_table(train, GUIDE_TERMS)

    for name in ("dev", "test"):
        df = splits[name]
        hit = df["text"].map(lambda s: any(w in curated.terms for w in words(s)))
        off = df["label"].isin([1, 2, 3, 4])
        lines += ["", f"## Curated lexicon on {name}", "",
                  f"- coverage {hit.mean():.1%}; precision for 'offensive' {off[hit].mean():.1%}; "
                  f"recall of offensive comments {hit[off].mean():.1%}"]
    (out_dir / "lexicon_report.md").write_text("\n".join(lines) + "\n")
    mine(train).head(300).to_csv(out_dir / "lexicon_candidates.tsv", sep="\t", index=False)
    print("\n".join(lines))


def build_translit(splits, data_dir: str, use_xlit: bool) -> None:
    path = Path(data_dir) / "translit_cache.tsv"
    cache = load_cache(path)
    texts = sorted({normalize(t) for df in splits.values() for t in df["text"]})
    xlit = XlitRomanToTamil() if use_xlit else None
    todo = [t for t in texts if t not in cache]
    print(f"transliteration: {len(texts)} unique texts, {len(todo)} not cached")
    for i, t in enumerate(todo, 1):
        if script_of(t) == "roman" and xlit is None:
            continue  # the rule-based fallback only covers Tamil -> roman
        view = alt_view(t, xlit)
        if view:
            cache[t] = view
        if i % 2000 == 0:
            save_cache(cache, path)
            print(f"  {i}/{len(todo)}")
    save_cache(cache, path)
    print(f"wrote {path} ({len(cache)} views)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--xlit", action="store_true", help="romanised -> Tamil with IndicXlit (needs fairseq)")
    args = ap.parse_args()
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    splits = load_splits(args.data_dir)
    lexicon_report(splits, out_dir)
    build_translit(splits, args.data_dir, args.xlit)


if __name__ == "__main__":
    main()
