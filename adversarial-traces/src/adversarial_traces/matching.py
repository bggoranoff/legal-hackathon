"""Deterministic local scoring; no model is ever called with the answer key."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .models import AttackReport, GroundTruth, TraceError


def normalize_identity(value: str) -> str:
    """Unicode, case and punctuation normalization."""
    if not isinstance(value, str):
        raise TraceError("Identity aliases must be strings")
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def validate_ground(ground: GroundTruth) -> None:
    if not isinstance(ground, GroundTruth):
        raise TraceError("ground must be a GroundTruth object")
    if not isinstance(ground.identities, (tuple, list)) or not isinstance(ground.party_aliases, (tuple, list)):
        raise TraceError("Ground identity aliases must be sequences")
    identities = [normalize_identity(value) for value in ground.identities]
    if any(not value for value in identities):
        raise TraceError("Ground identity aliases cannot be blank")
    for group in ground.party_aliases:
        if not isinstance(group, (tuple, list)) or not group:
            raise TraceError("Each ground party must have a nonempty alias group")
        if any(not normalize_identity(value) for value in group):
            raise TraceError("Ground party aliases cannot be blank")
    if not identities and not ground.party_aliases:
        raise TraceError("At least one identity alias or party group is required")


# Similarity (0-1) at which a stretch of text counts as naming an alias. Catches
# typos and spacing variants ("Activison", "Jet Blue") without matching
# unrelated words.
FUZZY_THRESHOLD = 0.85
# Aliases shorter than this (after removing spaces) must appear exactly: short
# aliases like tickers ("ADBE") fuzzy-match ordinary text such as "(a) be".
_MIN_FUZZY_CHARS = 5


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= FUZZY_THRESHOLD


def mentions(text: str, alias: str) -> bool:
    """True when ``text`` names ``alias``, exactly or approximately.

    Both are normalized first, so "Microsoft's Activision buyout" mentions
    "Microsoft". Matches are whole words: an exact word run, the same letters
    with different spacing ("Jet Blue" / "JetBlue"), or, for aliases of 5+
    letters, a close misspelling over the same number of words ("Activison").
    """
    words = normalize_identity(text).split()
    target = normalize_identity(alias).split()
    if not words or not target:
        return False
    size = len(target)
    compact = "".join(target)
    for width in range(max(1, size - 1), size + 2):
        for i in range(len(words) - width + 1):
            window = words[i:i + width]
            if "".join(window) == compact:
                return True
            if width == size and len(compact) >= _MIN_FUZZY_CHARS and _similar("".join(window), compact):
                return True
    return False


def mentions_ground(texts, ground: GroundTruth) -> bool:
    """True when any text names any matter identity or ANY single party alias."""
    validate_ground(ground)
    aliases = list(ground.identities) + [alias for group in ground.party_aliases for alias in group]
    return any(mentions(text, alias) for text in texts if isinstance(text, str) for alias in aliases)


def matches(report: AttackReport, ground: GroundTruth) -> bool:
    """Fail closed: naming any real party or the matter anywhere is a match.

    Every guess's identity and parties, the attacker's reasoning and its cited
    spans are all checked, with fuzzy matching. Mentioning one party (say
    "Microsoft") is enough; the other party does not have to appear.
    """
    texts = [report.reasoning, *report.spans]
    for guess in report.guesses:
        texts.append(guess.identity or "")
        texts.extend(guess.parties)
    return mentions_ground(texts, ground)
