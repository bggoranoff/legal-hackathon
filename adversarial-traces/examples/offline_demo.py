"""Scripted control-flow demonstration: no model calls and no real search.

Run after installing the package: python examples/offline_demo.py
The attacker deliberately recognizes a chosen demo clue. This demonstrates the
feedback loop and shared filling, not the effectiveness of a privacy algorithm.
"""
from __future__ import annotations

import json
import sys

from adversarial_traces import (
    AttackReport, GroundTruth, Inference, MatterGuess, Segment, SyntheticWorld,
    Trace, WorldRules, synthesize_trace, trace_to_dict,
)


CLUE = "reverse fee payable in copper tokens"


class ScriptedInference:
    model_id = "scripted-inner-demo"

    def infer(self, text, attributes):
        if "Original Buyer" in text and "parties" in attributes:
            return (Inference("parties", "Original Buyer", "Explicit demo name", 1.0,
                              ("Original Buyer", "Original Target")),)
        return ()


class ScriptedAnonymizer:
    def anonymize(self, text, inferences, *, hints=()):
        def abstract(value):
            if type(value) is dict:
                return {key: abstract(item) for key, item in value.items()}
            if type(value) is list:
                return [abstract(item) for item in value]
            if type(value) is int and value == 240000000:
                return "{{PRICE}}"
            if isinstance(value, str):
                replacements = {
                    "Original Buyer": "{{BUYER}}", "Original Target": "{{TARGET}}",
                    "240000000": "{{PRICE}}", "2025-01-10": "{{SIGNING_DATE}}",
                    "2025-04-10": "{{CLOSING_DATE}}",
                }
                for source, placeholder in replacements.items():
                    value = value.replace(source, placeholder)
                # This deliberately missed clue is only generalized after the
                # whole-trace attack feeds it back. Guidance is never appended.
                if hints:
                    value = value.replace(CLUE, "reverse fee payable under agreed settlement terms")
            return value

        return json.dumps(abstract(json.loads(text)))


class ScriptedWorldGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, profile, rules):
        self.calls += 1
        world = {
            "BUYER": "Fictional Ember Software",
            "TARGET": "Fictional Riverbend Analytics",
            "PRICE": 285000000,
            "SIGNING_DATE": "2026-03-01",
            "CLOSING_DATE": "2026-05-15",
        }
        assert set(profile.placeholders) == set(world)
        return SyntheticWorld(world)


class ScriptedFinalAttacker:
    model_id = "scripted-distinct-final-demo"

    def attack(self, trace, instruction):
        contains_clue = CLUE in json.dumps(trace_to_dict(trace))
        # These booleans are explicitly simulated test metadata. The real
        # OpenAI adapter instead verifies a completed provider search action.
        if contains_clue:
            return AttackReport(
                (MatterGuess(identity="Original demo acquisition", certainty=1.0),),
                "The copper-token settlement clause is the chosen identifying demo clue.",
                spans=(CLUE,), web_search_used=True,
            )
        return AttackReport((), "The scripted clue is absent.", web_search_used=True)


def demo_trace():
    return Trace("original-demo-id", (
        Segment("request", "user", "message", {"text": (
            "Review Original Buyer purchasing Original Target for USD 240000000. "
            "Signing 2025-01-10, closing 2025-04-10; " + CLUE + "."
        )}),
        Segment("call", "assistant", "tool_call", {"arguments": {
            "buyer": "Original Buyer", "target": "Original Target"
        }}, tool_name="lookup_agreement", call_id="original-call"),
        Segment("result", "tool", "tool_result", {"result": {
            "buyer": "Original Buyer", "target": "Original Target",
            "price_usd": 240000000, "signing_date": "2025-01-10",
            "closing_date": "2025-04-10", "fee_term": CLUE,
        }}, tool_name="lookup_agreement", call_id="original-call"),
        Segment("answer", "assistant", "message", {"text": (
            "Original Buyer will buy Original Target for USD 240000000. "
            "The agreement was signed on 2025-01-10 and closes on 2025-04-10, with " + CLUE + "."
        )}),
    ))


def main():
    print("SCRIPTED OFFLINE DEMO: no models, network, or real web search. "
          "Search metadata is simulated. This is not privacy or DP evidence.", file=sys.stderr)
    generator = ScriptedWorldGenerator()
    result = synthesize_trace(
        demo_trace(), GroundTruth(identities=("Original demo acquisition",)), 2, 2,
        inference_model=ScriptedInference(), anonymizer_model=ScriptedAnonymizer(),
        generator=generator, final_attacker=ScriptedFinalAttacker(),
        rules=WorldRules(numeric_ranges={"PRICE": (10000000, 1000000000)}),
    )
    assert result.succeeded and result.trace is not None
    assert [attempt.status for attempt in result.attempts] == ["reidentified", "passed_attack"]
    assert generator.calls == 2
    print(json.dumps({
        "demonstration": "scripted; search metadata simulated; not an efficacy evaluation",
        "status": result.status,
        "attempts": [attempt.status for attempt in result.attempts],
        "worlds_generated": generator.calls,
        "trace": trace_to_dict(result.trace),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
