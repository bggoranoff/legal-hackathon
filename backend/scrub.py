"""PII span detection and consistent-placeholder replacement.

Ported from TokenTrim/orizon-scrub. Detection is done by OpenAI's Privacy Filter
model via the official ``opf`` package (https://github.com/openai/privacy-filter),
run at a high-recall operating point. The model returns character-offset spans +
categories; we do our own replacement so the same entity maps to the same numbered
token within a conversation (PERSON_1, EMAIL_1, ...), which keeps traces analyzable.

The model's spans are augmented with high-precision regex spans for emails, credit
cards, and secret keys, which take precedence on overlap because the model is
unreliable on these structured types.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass

# Model category label -> friendly placeholder prefix. The v2 taxonomy shipped by
# openai/privacy-filter is exactly these eight labels.
CATEGORY_PREFIX = {
    "private_person": "PERSON",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "ADDRESS",
    "private_url": "URL",
    "private_date": "DATE",
    "account_number": "ACCOUNT",
    "secret": "SECRET",
}


def placeholder_prefix(category: str) -> str:
    """The token prefix used for a category in placeholders.

    ``private_email`` -> ``EMAIL`` (via the built-in map); an unmapped category
    like ``employee_id`` -> ``EMPLOYEE_ID``.
    """
    prefix = CATEGORY_PREFIX.get(category)
    if prefix is None:
        prefix = re.sub(r"[^A-Za-z0-9]+", "_", category.upper()).strip("_") or "PII"
    return prefix


# High-recall Viterbi calibration for the opf CRF decoder. The shipped calibration
# is all-zeros; opf validates this artifact has EXACTLY these keys (see
# opf._core.decoding.resolve_viterbi_biases_from_calibration_path). Directions come
# from opf._core.decoding._transition_bias: leaving/avoiding the background state and
# extending/chaining spans maximizes recall (we would rather over-redact than leak).
HIGH_RECALL_BIASES = {
    "transition_bias_background_stay": -2.5,
    "transition_bias_background_to_start": 2.5,
    "transition_bias_inside_to_continue": 1.0,
    "transition_bias_inside_to_end": 0.0,
    "transition_bias_end_to_background": -1.0,
    "transition_bias_end_to_start": 1.0,
}


@dataclass(frozen=True)
class Span:
    """A detected PII span: character offsets into the detector's base text."""

    start: int
    end: int
    category: str


class PrivacyFilterDetector:
    """Lazy, reusable wrapper around the ``opf`` model at a high-recall operating point.

    The model (~2.8GB) is downloaded and loaded on first ``detect`` call, then
    reused for the whole run. ``opf``'s ``OPF()`` constructor defaults to CUDA and
    crashes on CPU-only machines, so the device is always set explicitly.
    """

    def __init__(self, device: str | None = None, biases: dict | None = None):
        self._device = device
        self._biases = dict(biases or HIGH_RECALL_BIASES)
        self._redactor = None
        self._calib_path: str | None = None

    def _ensure_loaded(self) -> None:
        if self._redactor is not None:
            return
        try:
            import torch
            from opf import OPF
        except ImportError as exc:
            raise RuntimeError(
                "The 'opf' package (openai/privacy-filter) is required for scrubbing. "
                "Install it with:\n"
                "  pip install 'git+https://github.com/openai/privacy-filter.git'"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        fd, path = tempfile.mkstemp(prefix="orizon_scrub_viterbi_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"operating_points": {"default": {"biases": self._biases}}}, fh)
        self._calib_path = path
        redactor = OPF(
            output_mode="typed",
            decode_mode="viterbi",
            device=device,
            discard_overlapping_predicted_spans=False,
        )
        redactor.set_viterbi_decoder(calibration_path=path)
        self._redactor = redactor

    def detect(self, text: str) -> tuple[str, list[Span]]:
        """Return ``(base_text, spans)``.

        ``base_text`` is the string the spans index into. It equals ``text`` unless
        the tokenizer round-trip did not reproduce the input exactly, in which case
        opf's spans refer to the decoded text, so we return that to keep offsets valid.

        The model's spans are augmented with high-precision regex spans (email, card,
        secret); the deterministic patterns take precedence on overlap.
        """
        if not text:
            return text, []
        self._ensure_loaded()
        result = self._redactor.redact(text)
        base = result.text
        model_spans = [Span(s.start, s.end, s.label) for s in result.detected_spans]
        return base, merge_spans(regex_pii_spans(base), model_spans)


class Scrubber:
    """Replace detected PII with consistent, per-conversation numbered placeholders.

    One instance holds the alias table for a single conversation, so the same value
    maps to the same token (``[PERSON_1]``) across every string scrubbed with it.
    """

    def __init__(self, detector):
        self.detector = detector
        self._alias: dict[tuple[str, str], str] = {}
        self._per_cat: dict[str, int] = {}

    def scrub_text(self, text: str) -> str:
        """Return ``text`` with every detected PII span replaced by a numbered token."""
        if not isinstance(text, str) or not text:
            return text
        base, spans = self.detector.detect(text)
        valid = [sp for sp in spans if 0 <= sp.start < sp.end <= len(base)]
        if not valid:
            return base
        # Assign placeholders in document order so numbering reads 1, 2, 3 ...
        plan = [(sp, self._placeholder(sp.category, base[sp.start : sp.end]))
                for sp in sorted(valid, key=lambda s: s.start)]
        # Splice right-to-left so earlier offsets stay valid.
        out = base
        for sp, token in sorted(plan, key=lambda t: t[0].start, reverse=True):
            out = out[: sp.start] + token + out[sp.end :]
        return out

    def _placeholder(self, category: str, value: str) -> str:
        prefix = placeholder_prefix(category)
        # Normalize to alphanumerics so one entity written in different formats maps
        # to a single token; fall back to the raw form if it has no alphanumerics.
        norm = re.sub(r"[^a-z0-9]+", "", value.lower()) or value.strip().lower()
        key = (category, norm)
        token = self._alias.get(key)
        if token is None:
            self._per_cat[category] = self._per_cat.get(category, 0) + 1
            token = f"[{prefix}_{self._per_cat[category]}]"
            self._alias[key] = token
        return token


# ---------------------------------------------------------------------------
# High-precision structured-PII patterns: a detection supplement (the model is
# unreliable on these) whose spans take precedence over the model's on overlap.
# ---------------------------------------------------------------------------

# Bounded quantifiers keep this linear: an unbounded domain run before the TLD
# causes O(n^2) backtracking on pathological input, which would hang detection.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9][A-Za-z0-9.\-]{0,253}\.[A-Za-z]{2,24}")
# A run of >=13 digits, optionally single-separated by spaces/hyphens. We scan
# windows inside each run for a Luhn-valid card, so a card fused to adjacent digits
# is still found.
_DIGIT_RUN_RE = re.compile(r"\d(?:[ \-]?\d){12,}")
_SECRET_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:sk|rk|pk|api)[_\-](?:live|test|prod|proj)[_\-][A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _looks_like_identifier(text: str, start: int, end: int, run: str) -> bool:
    """True when the digit run at ``[start, end)`` is a fragment of an identifier
    (a hex trace id, a base64 token) rather than a card number.

    Only runs directly touching an ASCII letter are candidates: a card in prose or
    JSON is bounded by whitespace, quotes, or punctuation. A run written with
    internal separators (``4242 4242 ...``) is a formatted card, never an id. For a
    contiguous run glued to letters, inspect the enclosing alphanumeric token: if,
    after removing a single leading and trailing alphabetic run, letters remain
    interspersed among the digits, it is hex/base64 and skipped; a clean word wrapped
    around a pure digit block (``card4242...4242``) is left for the Luhn check.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if not (_is_ascii_letter(before) or _is_ascii_letter(after)):
        return False
    if any(c in " -" for c in run):
        return False
    left = start
    while left > 0 and text[left - 1].isascii() and text[left - 1].isalnum():
        left -= 1
    right = end
    while right < len(text) and text[right].isascii() and text[right].isalnum():
        right += 1
    token = text[left:right]
    i = 0
    while i < len(token) and _is_ascii_letter(token[i]):
        i += 1
    j = len(token)
    while j > i and _is_ascii_letter(token[j - 1]):
        j -= 1
    return any(_is_ascii_letter(c) for c in token[i:j])


def _card_char_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of Luhn-valid 13-19 digit card numbers within ``text``.

    Scans each maximal digit run for the longest, earliest Luhn-valid window so a
    card adjacent to other digits is still located precisely. Runs that are a
    fragment of an alphanumeric identifier are skipped (see
    :func:`_looks_like_identifier`).
    """
    spans: list[tuple[int, int]] = []
    for m in _DIGIT_RUN_RE.finditer(text):
        run = m.group()
        if _looks_like_identifier(text, m.start(), m.end(), run):
            continue
        idx = [i for i, ch in enumerate(run) if ch.isdigit()]
        digits = "".join(run[i] for i in idx)
        n = len(digits)
        for length in range(min(19, n), 12, -1):
            hit = None
            for start in range(n - length + 1):
                if _luhn_ok(digits[start : start + length]):
                    hit = (m.start() + idx[start], m.start() + idx[start + length - 1] + 1)
                    break
            if hit:
                spans.append(hit)
                break
    return spans


def merge_spans(primary: list[Span], secondary: list[Span]) -> list[Span]:
    """Merge two span lists into a non-overlapping set; ``primary`` wins on overlap.

    Within each list, longer spans win. Used to give the deterministic regex spans
    precedence over the model's (which can mislabel structured PII).
    """
    kept: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for group in (primary, secondary):  # primary placed first => wins on overlap
        for sp in sorted(group, key=lambda s: (-(s.end - s.start), s.start)):
            if any(sp.start < e and sp.end > s for s, e in occupied):
                continue
            occupied.append((sp.start, sp.end))
            kept.append(sp)
    return kept


def regex_pii_spans(text: str) -> list[Span]:
    """High-precision spans for emails, cards (Luhn-checked), and secrets."""
    spans: list[Span] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append(Span(m.start(), m.end(), "private_email"))
    for rx in _SECRET_RES:
        for m in rx.finditer(text):
            spans.append(Span(m.start(), m.end(), "secret"))
    for start, end in _card_char_spans(text):
        spans.append(Span(start, end, "account_number"))
    return merge_spans(spans, [])
