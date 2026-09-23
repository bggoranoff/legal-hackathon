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
# Aliases shorter than this (after removing spaces) must appear exactly.
_MIN_FUZZY_CHARS = 4


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= FUZZY_THRESHOLD


def mentions(text: str, alias: str) -> bool:
    """True when ``text`` names ``alias``, exactly or approximately.

    Both are normalized first, so "Microsoft's Activision buyout" mentions
    "Microsoft". Whole words must line up: "Microsoftware" does not mention
    "Microsoft" exactly, though a close misspelling of it does.
    """
    words = normalize_identity(text).split()
    target = normalize_identity(alias).split()
    if not words or not target:
        return False
    size = len(target)
    for i in range(len(words) - size + 1):
        if words[i:i + size] == target:
            return True
    compact = "".join(target)
    if len(compact) < _MIN_FUZZY_CHARS:
        return False
    # Compare against runs of one fewer to one more words so spacing
    # differences ("Jet Blue" vs "JetBlue") still line up.
    for width in range(max(1, size - 1), size + 2):
        for i in range(len(words) - width + 1):
            if _similar("".join(words[i:i + width]), compact):
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
