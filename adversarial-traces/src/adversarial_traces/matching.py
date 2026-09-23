"""Deterministic scoring; no model is ever called with the answer key."""
from __future__ import annotations

import re
import unicodedata

from .models import AttackReport, GroundTruth, TraceError


def normalize_identity(value: str) -> str:
    """Case/punctuation normalization, deliberately without fuzzy matching."""
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


def matches(report: AttackReport, ground: GroundTruth) -> bool:
    """Any candidate matching an identity OR every required party counts.

    Aliases are explicit. A partial word/substring match is never sufficient.
    Parties must occur together in one candidate, and a single named party
    cannot satisfy two ground-truth party groups.
    """
    validate_ground(ground)
    identities = {normalize_identity(value) for value in ground.identities}
    groups = [{normalize_identity(value) for value in group} for group in ground.party_aliases]
    for guess in report.guesses:
        if guess.identity is not None and normalize_identity(guess.identity) in identities:
            return True
        names = {normalize_identity(value) for value in guess.parties if normalize_identity(value)}
        if not groups or len(names) < len(groups):
            continue
        assigned: dict[str, int] = {}

        def assign(group_index: int, seen: set[str]) -> bool:
            for name in sorted(groups[group_index] & names):
                if name in seen:
                    continue
                seen.add(name)
                if name not in assigned or assign(assigned[name], seen):
                    assigned[name] = group_index
                    return True
            return False

        if all(assign(index, set()) for index in range(len(groups))):
            return True
    return False
