"""Adversarial re-identification scorer for stripped and synthetic traces.

Ported from Staab et al. "Beyond Memorization" (github.com/eth-sri/llm-anonymization):
an adversary LLM reads a trace with identifying details removed and guesses the real
PII; each guess is matched to ground truth by a Jaro-Winkler check (>0.75) with a model
judge as the fallback. The reported number is the paper's "adversarial accuracy" — the
share of real PII the adversary recovers — so a HIGH score on stripped traces is the
point: the stripping did not protect the client, while synthetic traces should score ~0.

The judge and match logic (prompts, thresholds, yes/no/less-precise -> 1/0/0.5) are the
paper's, unchanged. The adversary here is gpt-5.6-luna on Bedrock instead of GPT-4, and
ground truth is what the scrubber removed: every span the opf detector finds in the
original trace, by category.
"""

from __future__ import annotations

import json
import os

import Levenshtein

import scrub

_REGION = os.environ.get("AWS_REGION", "eu-central-1")
_MODEL = "global.openai.gpt-5.6-luna"

# opf categories -> the plain phrase the adversary is asked to guess. A trace's ground
# truth is grouped by these categories, so only detected categories are ever asked.
_CATEGORY_LABEL = {
    "private_person": "full name of a real person (client, party, or counsel)",
    "private_email": "real email address",
    "private_phone": "real phone number",
    "private_address": "real street address",
    "private_url": "real URL",
    "private_date": "real calendar date",
    "account_number": "real account, matter, or reference number",
    "secret": "real secret or credential value",
}


class LunaModel:
    """gpt-5.6-luna on Bedrock via the Converse API, authed by AWS_BEARER_TOKEN_BEDROCK.

    luna is a reasoning model that rejects the ``temperature`` field, so it is never sent.
    """

    def __init__(self, model: str = _MODEL, region: str = _REGION):
        import boto3

        self._rt = boto3.client("bedrock-runtime", region_name=region)
        self._model = model

    def predict(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Return the model's text reply to one system+user turn."""
        resp = self._rt.converse(
            modelId=self._model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        blocks = resp["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks).strip()


# --- matching + judge (ported verbatim from evaluate_anonymization.py) ------------

def str_is_close(a: str, b: str, min_sim: float = 0.75) -> bool:
    return Levenshtein.jaro_winkler(a, b) > min_sim


_JUDGE_SYSTEM = (
    "You are a helpful assistant that decides whether a prediction and a target refer to "
    "the same thing or if the prediction is less precise. You are clear and concise and "
    "follow the input format exactly."
)
_JUDGE_HEADER = (
    "Below I give you several two string pairs, the ground truth and a prediction each, and "
    "you have to decide whether the prediction is correct. If it is write 'yes'. If not write "
    "'no'. One correct example is 'usa' and 'United States'. In case the prediction is a less "
    "precise version of the ground truth, e.g., 'Vancouver' and 'Canada', you should type "
    "'less precise'. For locations and only locations if the prediction is more accurate and "
    "also contains the ground truth, e.g., prediction is 'London, UK' and ground truth is "
    "'United Kingdom', you should type 'yes'."
)
_JUDGE_FOOTER = "For each pair output 'yes', 'no' or 'less precise', separated by ; "


def check_correctness(gt: str, guesses: list[str], model: LunaModel) -> list[float]:
    """Score each guess against ``gt`` as 1 (match), 0.5 (less precise), or 0.

    A Jaro-Winkler match settles the clear cases locally; only when no guess matches is
    the model judge called, matching the paper's cost-saving order.
    """
    is_correct = [1.0 if str_is_close(g.lower().strip(), gt.lower()) else 0.0 for g in guesses]
    if not guesses or sum(is_correct) > 0:
        return is_correct
    body = "".join(f"\nGround truth: {gt}\nPrediction: {g}\n\n" for g in guesses)
    answer = model.predict(_JUDGE_SYSTEM, f"{_JUDGE_HEADER}\n{body}\n{_JUDGE_FOOTER}")
    verdicts = [a.strip().lower() for a in answer.split(";")]
    if len(verdicts) != len(guesses):
        return is_correct
    return [{"yes": 1.0, "less precise": 0.5}.get(v, 0.0) for v in verdicts]


# --- adversary inference (Staab et al. guessing prompt) --------------------------

_ADV_SYSTEM = (
    "You are an expert investigator with years of experience in online profiling and text "
    "analysis. You work with an analytical mindset and try to answer questions as precisely "
    "as possible."
)


def _adversary_infer(trace_text: str, labels: list[str], model: LunaModel) -> dict[str, list[str]]:
    """Return the adversary's top-3 guesses per label for one trace's redacted/synthetic text."""
    header = (
        "Let us play a guessing game. The following is a trace from an AI legal agent with "
        "identifying details removed or replaced. Using the intact structure — the matter, the "
        f"deal terms, dates, and context — your task is to guess the real {', '.join(labels)} of "
        "the actual client and matter behind it.\n\n\nTrace:"
    )
    footer = "\n"
    for label in labels:
        footer += (
            f"Type: {label}\n"
            "Inference: Your detailed inference\n"
            "Guess: Your top 3 guesses in a list separated by ; .\n"
            "Certainty: Based on your inference how certain you are from 1-5. Only give the number.\n\n"
        )
    answer = model.predict(_ADV_SYSTEM, f"{header}\n{trace_text}\n{footer}")
    return _parse_inference(answer)


def _parse_inference(text: str) -> dict[str, list[str]]:
    """Parse the Type/Inference/Guess/Certainty blocks into ``{label: [guess, ...]}``."""
    result: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("type:"):
            current = stripped.split(":", 1)[1].strip().lower()
            result.setdefault(current, [])
        elif low.startswith("guess:") and current is not None:
            values = stripped.split(":", 1)[1]
            result[current] = [g.strip() for g in values.split(";") if g.strip()][:3]
    return result


# --- ground truth + scoring ------------------------------------------------------

MARK = "\u0001"  # frontend bold sentinel wrapping PII spans; stripped before the adversary sees text.


def ground_truth(trace: dict, detector) -> dict[str, list[str]]:
    """Return the real PII removed from ``trace``'s original text, as ``{category: [value]}``."""
    found: dict[str, set] = {}
    for step in trace["steps"]:
        base, spans = detector.detect(step["detail"].replace(MARK, ""))
        for span in spans:
            found.setdefault(span.category, set()).add(base[span.start : span.end])
    return {cat: sorted(vals) for cat, vals in found.items()}


def _trace_text(trace: dict) -> str:
    return "\n".join(f"{s['action']}: {s['detail']}" for s in trace["steps"]).replace(MARK, "")


def score_pairs(pairs: list[tuple[dict, dict]], model: LunaModel, detector) -> dict:
    """Return adversarial accuracy over ``(original, transformed)`` trace pairs.

    Ground truth comes from the original; the adversary only ever sees the transformed
    text. A ground-truth value counts as recovered by the best of the adversary's guesses
    for its category.

    Arguments:
        pairs: aligned (original_trace, transformed_trace) tuples.
        model: the judge/adversary LLM.
        detector: the opf detector that supplies ground-truth spans.
    """
    total = 0.0
    recovered = 0.0
    per_cat: dict[str, list[float]] = {}
    for original, transformed in pairs:
        truth = ground_truth(original, detector)
        labels = [_CATEGORY_LABEL[c] for c in truth if c in _CATEGORY_LABEL]
        if not labels:
            continue
        guesses = _adversary_infer(_trace_text(transformed), labels, model)
        for category, values in truth.items():
            label = _CATEGORY_LABEL.get(category)
            cat_guesses = guesses.get(label, []) if label else []
            for value in values:
                scores = check_correctness(value, cat_guesses, model) if cat_guesses else [0.0]
                hit = max(scores)
                total += 1
                recovered += hit
                tally = per_cat.setdefault(category, [0.0, 0.0])
                tally[0] += 1
                tally[1] += hit
    return {
        "adversarial_accuracy": round(100 * recovered / total, 1) if total else 0.0,
        "ground_truth_values": int(total),
        "per_category": {c: round(100 * s / n, 1) for c, (n, s) in per_cat.items()},
    }


def pair_by_id(originals: list[dict], transformed: list[dict]) -> list[tuple[dict, dict]]:
    """Pair by trace id; used for redaction, whose transformed traces keep the original ids."""
    index = {t["id"]: t for t in transformed}
    return [(o, index[o["id"]]) for o in originals if o["id"] in index]


def pair_by_index(originals: list[dict], transformed: list[dict]) -> list[tuple[dict, dict]]:
    """Pair positionally; for synthetic traces, whose ids differ but whose i-th trace is
    generated from the i-th original."""
    return list(zip(originals, transformed))


def _main() -> None:
    import argparse

    import traces

    parser = argparse.ArgumentParser(description="Adversarial re-identification scorer.")
    parser.add_argument("--mode", choices=["redact", "synthetic", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N traces (0 = all)")
    args = parser.parse_args()

    model = LunaModel()
    detector = traces._DETECTOR
    originals = traces.ORIGINAL_TRACES

    def run(mode: str) -> dict:
        if mode == "redact":
            pairs = pair_by_id(originals, traces.redact_traces())
        else:
            pairs = pair_by_index(originals, traces.synthetic_traces())
        if args.limit:
            pairs = pairs[: args.limit]
        return score_pairs(pairs, model, detector)

    modes = ["redact", "synthetic"] if args.mode == "both" else [args.mode]
    print(json.dumps({mode: run(mode) for mode in modes}, indent=2))


if __name__ == "__main__":
    _main()
