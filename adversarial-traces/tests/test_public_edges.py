"""Public API edge cases affecting valid inputs or false acceptance."""
import json
import unittest
from types import SimpleNamespace

from adversarial_traces import (
    AttackReport, GroundTruth, Inference, Segment, SyntheticWorld, Trace,
    TraceError, adversarial_anonymize, synthesize_trace,
)


class PublicEdgeTests(unittest.TestCase):
    def test_attribute_set_is_supported(self):
        calls = []

        def infer(text, attributes):
            calls.append(attributes)
            return ()

        result = adversarial_anonymize(
            "example", {"parties", "dates"}, 2,
            inference_model=SimpleNamespace(model_id="inner", infer=infer),
            anonymizer_model=SimpleNamespace(anonymize=lambda *args, **kwargs: self.fail("Unexpected rewrite")),
        )
        self.assertEqual(result, "example")
        self.assertEqual(calls, [("dates", "parties")])

    def test_candidate_hook_can_reject_before_final_attack(self):
        def reject(_trace):
            raise TraceError("Client-side semantic constraint failed")

        result = synthesize_trace(
            Trace("source", (Segment("one", "user", "message", {"text": "generic text"}),)),
            GroundTruth(identities=("original matter",)), 1, 1,
            inference_model=SimpleNamespace(model_id="inner", infer=lambda *args: ()),
            anonymizer_model=SimpleNamespace(anonymize=lambda *args, **kwargs: self.fail("Unexpected rewrite")),
            generator=SimpleNamespace(generate=lambda *args: SyntheticWorld({})),
            final_attacker=SimpleNamespace(model_id="final", attack=lambda *args: self.fail("Rejected candidate attacked")),
            candidate_validator=reject,
        )
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.trace)

    def test_default_tool_schema_rejects_fractional_search_limit(self):
        trace = Trace("source", (
            Segment("call", "assistant", "tool_call", {"arguments": {
                "case_id": "sample", "query": "a clause", "limit": 3,
            }}, "search_documents", "call1"),
            Segment("result", "tool", "tool_result", {"result": {"results": []}},
                    "search_documents", "call1"),
        ))

        def infer(text, attributes):
            if '"limit": 3' in text:
                return (Inference("amounts", 3, certainty=1),)
            return ()

        def anonymize(text, *_args, **_kwargs):
            payload = json.loads(text)
            payload["arguments"]["limit"] = "{{LIMIT}}"
            return json.dumps(payload)

        result = synthesize_trace(
            trace, GroundTruth(identities=("original matter",)), 1, 1,
            inference_model=SimpleNamespace(model_id="inner", infer=infer),
            anonymizer_model=SimpleNamespace(anonymize=anonymize),
            generator=SimpleNamespace(generate=lambda *args: SyntheticWorld({"LIMIT": 2.5})),
            final_attacker=SimpleNamespace(model_id="final", attack=lambda *args: self.fail("Invalid tool call attacked")),
        )
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.trace)


if __name__ == "__main__":
    unittest.main()
