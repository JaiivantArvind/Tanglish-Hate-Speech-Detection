"""Text normalisation for Tanglish YouTube comments.

Deliberately light: the backbones are cased and have seen noisy web text, so
case is kept and only noise that cannot help the model is removed.
"""

from __future__ import annotations

import re

try:
    import emoji
except ImportError:  # optional dependency
    emoji = None

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACE = re.compile(r"\s+")


def _emoji_words(chars: str, data: dict) -> str:
    return " " + data.get("en", "").strip(":").replace("_", " ") + " "


def normalize(text: str, demojize: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    # Mentions mark a targeted comment, so keep a placeholder instead of deleting them.
    text = _MENTION.sub("@user", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _REPEAT.sub(r"\1\1", text)  # loooosu -> loosu, !!!!! -> !!, 😂😂😂 -> 😂😂
    if demojize and emoji is not None:
        # MuRIL and mBERT map every common emoji to [UNK]; their names carry the
        # sentiment. Plain words ("face with tears of joy") survive every tokenizer.
        text = emoji.replace_emoji(text, replace=_emoji_words)
    return _SPACE.sub(" ", text).strip()


_WORD = re.compile(r"[a-z]+|[\u0B80-\u0BFF]+")


def words(text: str) -> list[str]:
    """Lower-cased Latin or Tamil-script words, used for lexicon mining and matching."""
    return _WORD.findall(text.lower())
