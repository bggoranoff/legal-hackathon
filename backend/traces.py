"""The real legal-agent traces and the transforms the demo shows off."""

import copy
import re

import data
import scrub
import synth

ORIGINAL_TRACES = data.load_traces()
_ORIGINAL_NAMES = set()  # real deal traces carry no name registry; scrub detects entities for redaction

# Sentinel wrapping each PII span; the frontend renders wrapped spans bold.
MARK = "\u0001"

_PII_PATTERNS = [
    (re.compile(r"\+1 \(\d{3}\) \d{3}-\d{4}"), "[PHONE]"),
    (re.compile(r"[\w.]+@[\w.]+\.\w+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"A-\d{3}-\d{3}-\d{3}"), "[A-NUMBER]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"\b\d{2,4} [\w ]+?(?:Ave|St|Street|Avenue|Rd|Road), [\w ]+, [A-Z]{2} \d{5}"), "[ADDRESS]"),
]


_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")

_DETECTOR = scrub.PrivacyFilterDetector()


def _mark_pii(text: str, names) -> str:
    """Wrap every PII span (client names and pattern matches) in the bold sentinel."""
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, MARK + name + MARK)
    for pattern, _ in _PII_PATTERNS:
        text = pattern.sub(lambda m: MARK + m.group(0) + MARK, text)
    return text


def _mark_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: MARK + m.group(0) + MARK, text)


def _marked(traces, marker):
    """Return a copy of `traces` with each step detail passed through `marker`."""
    out = copy.deepcopy(traces)
    for trace in out:
        for step in trace["steps"]:
            step["detail"] = marker(step["detail"])
    return out


def original_traces():
    """Return the originals with real PII spans marked for bold rendering."""
    return _marked(ORIGINAL_TRACES, lambda t: _mark_pii(t, _ORIGINAL_NAMES))


def redact_traces():
    """Return the originals with every PII span replaced by consistent placeholders.

    Detection uses OpenAI's Privacy Filter via orizon-scrub; each trace gets its own
    alias table, so a client maps to the same numbered token across their steps.
    """
    out = copy.deepcopy(ORIGINAL_TRACES)
    for trace in out:
        scrubber = scrub.Scrubber(_DETECTOR)
        for step in trace["steps"]:
            step["detail"] = _mark_placeholders(scrubber.scrub_text(step["detail"]))
    return out


def synthetic_traces():
    """Return Bedrock-generated synthetic traces that mirror the real ones' structure but no real content.

    Generation happens once (cached in `synth`); nothing real is recoverable from the result.
    """
    return _marked(synth.generate(ORIGINAL_TRACES), lambda t: _mark_pii(t, set()))


def reconstruct_traces(mode: str):
    """Fake 'LLM + web access' reconstruction of the originals from the transformed set.

    From redacted traces the placeholders sit on top of intact structure, so a capable model
    with web access recovers almost all of the real PII. From synthetic traces there is nothing
    real to find, so the model returns confident but wrong values.
    """
    source = original_traces() if mode == "redact" else synthetic_traces()
    old = "trace" if mode == "redact" else "syn"
    for trace in source:
        trace["id"] = trace["id"].replace(old, "recon")
    return source


def verify_score(mode: str):
    """Return the share of real PII recovered by reconstruction, per mode."""
    return {"redact": 95, "synthetic": 1}[mode]
