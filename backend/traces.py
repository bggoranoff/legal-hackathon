"""The real legal-agent traces and the transforms the demo shows off."""

import concurrent.futures
import copy
import json
import os
import re

import data
import scrub
import synth

_SUFFIX = os.environ.get("CACHE_SUFFIX", "")  # set to e.g. ".full" to build caches off to the side
_REDACT_CACHE = os.path.join(os.path.dirname(__file__), "data", f"redacted_traces{_SUFFIX}.json")
_INFER_CACHE = os.path.join(os.path.dirname(__file__), "data", f"inferred_{{}}{_SUFFIX}.json")
_LIMIT = int(os.environ.get("DEMO_LIMIT", "10"))  # traces processed per transform, for a tractable demo

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


def _orgs(trace: dict) -> list:
    """The deal's party companies, from the matter label (e.g. ['Adobe', 'Figma'])."""
    return trace["matter"].split(" — ")[0].split(" / ")


def _strip_orgs(text: str, orgs: list) -> str:
    """Replace each party company (and its lead word) with a numbered [ORG_n] token."""
    for i, org in enumerate(orgs, 1):
        for variant in {org, org.split()[0]}:
            text = re.sub(rf"\b{re.escape(variant)}\b", f"[ORG_{i}]", text, flags=re.IGNORECASE)
    return text


def _run_scrub():
    out = copy.deepcopy(ORIGINAL_TRACES[:_LIMIT])
    for trace in out:
        scrubber = scrub.Scrubber(_DETECTOR)
        orgs = _orgs(trace)
        for step in trace["steps"]:
            step["detail"] = _strip_orgs(scrubber.scrub_text(step["detail"]), orgs)
    return out


def redact_traces():
    """Return the originals with every PII span replaced by consistent placeholders.

    Detection uses OpenAI's Privacy Filter via orizon-scrub; each trace gets its own
    alias table, so a client maps to the same numbered token across their steps. The scrub
    pass is run once and cached to `data/redacted_traces.json` so the demo stays fast.
    """
    if os.path.exists(_REDACT_CACHE):
        with open(_REDACT_CACHE) as f:
            scrubbed = json.load(f)
    else:
        scrubbed = _run_scrub()
        with open(_REDACT_CACHE, "w") as f:
            json.dump(scrubbed, f)
    return _marked(scrubbed, _mark_placeholders)


def synthetic_traces():
    """Return Bedrock-generated synthetic traces that mirror the real ones' structure but no real content.

    Generation happens once (cached in `synth`); nothing real is recoverable from the result.
    """
    return _marked(synth.generate(ORIGINAL_TRACES), lambda t: _mark_pii(t, set()))


_TOKEN_RE = re.compile(r"\[([A-Z]+)_(\d+)\]")

# The inferable identifier in an M&A matter: which deal (and thus which client) the trace is about.
_ADV_SYSTEM = (
    "You are an expert investigator who re-identifies parties from anonymized documents. "
    "You reason from context and answer precisely, in the exact format requested."
)


def _infer_attrs(trace_text: str, model):
    """Ask the adversary for the real companies and the deal's announcement year; return (orgs, year)."""
    prompt = (
        "The following trace is from an AI legal agent working an M&A matter, with the party names "
        "removed (shown as [ORG_1], [ORG_2], …). From the deal terms, sector, figures, dates and "
        "document context, identify the real-world companies involved AND the year the deal was "
        'announced. Reply with ONLY JSON: {"companies": ["A", "B"], "year": "YYYY"}, most confident '
        'company first, year null if unknown. No prose.\n\nTrace:\n' + trace_text
    )
    raw = model.predict(_ADV_SYSTEM, prompt)
    start, end = raw.find("{"), raw.rfind("}")
    try:
        obj = json.loads(raw[start:end + 1])
        orgs = [str(x) for x in obj.get("companies", []) if x]
        year = str(obj["year"]) if obj.get("year") else None
        return orgs, year
    except (ValueError, TypeError, AttributeError):
        return [], None


def _deal_year(original: dict):
    """The deal's year, taken from the source trace id (e.g. adobe_figma_2022_t01 -> '2022')."""
    match = re.search(r"(?:19|20)\d{2}", original["id"])
    return match.group(0) if match else None


def _fill_detail(text: str, org_guesses: list) -> str:
    """Replace each [ORG_n] token with the adversary's n-th inferred company (already bold-wrapped)."""
    if not org_guesses:
        return text
    def repl(m):
        if m.group(1) != "ORG":
            return m.group(0)
        idx = int(m.group(2)) - 1
        return org_guesses[idx] if idx < len(org_guesses) else org_guesses[0]
    return _TOKEN_RE.sub(repl, text)


def _infer(mode: str) -> dict:
    """Run the adversarial re-identification attack once and cache {reconstructed, score}.

    The attacker (score.py, per Staab et al.) sees only the transformed trace and guesses which
    deal — and thus which client — it is about. We fill the redacted [ORG] blanks with that guess
    (the reconstruction) and score recovery. Redaction leaks the deal identity (high score);
    synthetic traces carry no real deal, so nothing is recovered (low score).
    """
    path = _INFER_CACHE.format(mode)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    import score

    if mode == "redact":
        pairs = score.pair_by_id(ORIGINAL_TRACES, redact_traces())[:_LIMIT]
    else:
        by_id = {t["id"]: t for t in ORIGINAL_TRACES}
        pairs = [(by_id[t["source_id"]], t) for t in synthetic_traces() if t.get("source_id") in by_id]
    model = score.LunaModel()

    def _score_pair(pair):
        original, trans = pair
        orgs = _orgs(original)
        year = _deal_year(original)
        org_guesses, year_guess = _infer_attrs(score._trace_text(trans), model)
        hits = [max(score.check_correctness(v, org_guesses, model)) if org_guesses else 0.0 for v in orgs]
        if year:
            hits.append(1.0 if year_guess and year_guess.strip() == year else 0.0)
        recon = {
            "id": "recon-" + trans["id"],
            "matter": trans["matter"],
            "score": round(100 * sum(hits) / len(hits)) if hits else 0,
            "steps": [{"step": s["step"], "action": s["action"], "detail": _fill_detail(s["detail"], org_guesses)}
                      for s in trans["steps"]],
        }
        return recon, sum(hits), len(hits)

    # The attack is one independent Bedrock call per trace, so fan them out concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        scored = list(pool.map(_score_pair, pairs))

    reconstructed = [row[0] for row in scored]
    recovered = sum(row[1] for row in scored)
    total = sum(row[2] for row in scored)
    result = {"reconstructed": reconstructed, "score": round(100 * recovered / total, 1) if total else 0}
    with open(path, "w") as f:
        json.dump(result, f)
    return result


def reconstruct_traces(mode: str):
    """Return the attacker's reconstruction of the originals from the transformed traces."""
    return _infer(mode)["reconstructed"]


def verify_score(mode: str):
    """Return the percent of real PII the attacker recovered from the transformed traces."""
    return _infer(mode)["score"]
