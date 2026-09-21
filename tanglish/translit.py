"""Script views for code-mixed comments.

About 19% of comments are in Tamil script and the rest are romanised. Giving the
model the same comment in the other script helps it tie the two together.

* Tamil script -> romanised: a rule-based mapper that follows how people type
  Tanglish (கோமாளி -> komaali), always available.
* Romanised -> Tamil script: AI4Bharat IndicXlit, optional because it depends on
  fairseq. Run `python scripts/build_translit.py --xlit` where it installs
  (e.g. a Python 3.10 environment) and ship the cache file to the training machine.

Views are cached in a TSV (text -> alt view) so training never transliterates.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_VOWELS = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
}
_SIGNS = {
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e",
    "ே": "e", "ை": "ai", "ொ": "o", "ோ": "o", "ௌ": "au",
}
_CONSONANTS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "d", "ண": "n", "த": "th",
    "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v",
    "ழ": "zh", "ள": "l", "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h",
}
_VIRAMA = "்"
# Clusters that Tanglish writers spell differently from the letter-by-letter reading.
_CLUSTERS = {"ற்ற": "tr", "ன்ற": "ndr", "ச்ச": "ch"}
_TAMIL = re.compile(r"[\u0B80-\u0BFF]")
_LATIN = re.compile(r"[A-Za-z]")


def tamil_to_roman(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text[i : i + 3] in _CLUSTERS:
            cons, i = _CLUSTERS[text[i : i + 3]], i + 3
        elif text[i] in _CONSONANTS:
            cons, i = _CONSONANTS[text[i]], i + 1
        else:
            ch = text[i]
            i += 1
            if ch in _VOWELS:
                out.append(_VOWELS[ch])
            elif ch == "ஃ":
                out.append("h")
            elif ch not in _SIGNS and ch != _VIRAMA:  # drop stray signs
                out.append(ch)
            continue

        # A consonant (or cluster) takes a vowel sign, a virama, or the inherent "a".
        nxt = text[i] if i < len(text) else ""
        if nxt == _VIRAMA:
            out.append(cons)
            i += 1
        elif nxt in _SIGNS:
            out.append(cons + _SIGNS[nxt])
            i += 1
        else:
            out.append(cons + "a")
    return "".join(out)


def script_of(text: str) -> str:
    tamil = len(_TAMIL.findall(text))
    latin = len(_LATIN.findall(text))
    if tamil == 0 and latin == 0:
        return "other"
    return "tamil" if tamil >= latin else "roman"


class XlitRomanToTamil:
    """Wrapper around AI4Bharat IndicXlit (pip install ai4bharat-transliteration)."""

    def __init__(self, beam_width: int = 4):
        from ai4bharat.transliteration import XlitEngine  # noqa: deferred optional import

        self.engine = XlitEngine("ta", beam_width=beam_width, rescore=False)

    def __call__(self, text: str) -> str:
        return self.engine.translit_sentence(text)["ta"]


def alt_view(text: str, roman_to_tamil=None) -> str | None:
    """Returns the comment in the other script, or None if no view is available."""
    script = script_of(text)
    if script == "tamil":
        return tamil_to_roman(text)
    if script == "roman" and roman_to_tamil is not None:
        return roman_to_tamil(text)
    return None


def load_cache(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(row) == 2}


def save_cache(cache: dict[str, str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for src, dst in cache.items():
            f.write(f"{src.replace(chr(9), ' ')}\t{dst.replace(chr(9), ' ')}\n")
