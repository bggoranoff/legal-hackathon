"""Deterministic control-flow tests, not evidence of anonymization efficacy."""
from __future__ import annotations

import copy
import json
import unittest

from adversarial_traces.core import adversarial_anonymize, synthesize_trace
from adversarial_traces.models import (
    AttackReport,
    GroundTruth,
    Hint,
    Inference,
    MatterGuess,
    ModelResponseError,
    Segment,
    SyntheticWorld,
    Trace,
    TraceError,
)


class RecordingInference:
    model_id = "inner-test-model"

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or (lambda text, attributes: ())

    def infer(self, text, attributes):
        self.calls.append((text, attributes))
        return self.answer(text, attributes)


class RecordingAnonymizer:
    def __init__(self, rewrite=None):
        self.calls = []
        self.rewrite = rewrite or (lambda text, inferences, hints: text)

    def anonymize(self, text, inferences, *, hints=()):
        self.calls.append((text, inferences, hints))
        return self.rewrite(text, inferences, hints)


class RecordingGenerator:
    def __init__(self, make_world=None):
        self.calls = []
        self.make_world = make_world or (
            lambda profile, rules, n: SyntheticWorld({"BUYER": f"Invented Buyer {n}"})
        )

    def generate(self, profile, rules):
        self.calls.append((profile, rules))
        return self.make_world(profile, rules, len(self.calls))


class RecordingAttacker:
    model_id = "distinct-final-test-model"

    def __init__(self, reports=None):
        self.calls = []
        self.reports = reports or [AttackReport((), "No candidate found", web_search_used=True)]

    def attack(self, trace, instruction):
        self.calls.append((trace, instruction))
        report = self.reports[min(len(self.calls) - 1, len(self.reports) - 1)]
        if isinstance(report, Exception):
            raise report
        return report


def simple_trace():
    return Trace("sensitive-original-trace-id", (
        Segment("sensitive-original-segment-id", "user", "message", {
            "text": "Review Original Buyer buying the business."
        }),
    ))


def tool_trace():
    return Trace("original-trace-id", (
        Segment("s1", "user", "message", {"text": "Review Original Buyer."}),
        Segment("s2", "assistant", "tool_call", {
            "arguments": {"query": "Original Buyer merger", "limit": 2}
        }, tool_name="search", call_id="original-call-id"),
        Segment("s3", "tool", "tool_result", {
            "result": {"buyer": "Original Buyer", "documents": ["Agreement for Original Buyer"]}
        }, tool_name="search", call_id="original-call-id"),
        Segment("s4", "assistant", "message", {"text": "Original Buyer is the purchaser."}),
    ))


def pipeline_doubles():
    inference = RecordingInference(lambda text, attrs: (
        Inference("parties", "Original Buyer", "Explicit name", 1.0, ("Original Buyer",)),
    ) if "Original Buyer" in text else ())
    anonymizer = RecordingAnonymizer(
        lambda text, inferences, hints: text.replace("Original Buyer", "{{BUYER}}")
    )
    return inference, anonymizer, RecordingGenerator(), RecordingAttacker()


def run_pipeline(trace=None, ground=None, rounds=1, doubles=None):
    inference, anonymizer, generator, attacker = doubles or pipeline_doubles()
    result = synthesize_trace(
        trace or simple_trace(),
        ground or GroundTruth(identities=("SCORING_ONLY_IDENTITY_SENTINEL",)),
        1, rounds,
        inference_model=inference,
        anonymizer_model=anonymizer,
        generator=generator,
        final_attacker=attacker,
    )
    return result, (inference, anonymizer, generator, attacker)


class AdversarialAnonymizeTests(unittest.TestCase):
    def test_attacker_sees_only_current_text_and_stops_on_empty(self):
        inference = RecordingInference(lambda text, attrs: () if text == "version-2" else (
            Inference("parties", "Identifiable Party", certainty=0.9),
        ))
        anonymizer = RecordingAnonymizer(
            lambda text, findings, hints: "version-" + str(int(text.rsplit("-", 1)[1]) + 1)
        )
        source = "version-0"
        result = adversarial_anonymize(source, ("parties",), 10,
            inference_model=inference, anonymizer_model=anonymizer)
        self.assertEqual("version-2", result)
        self.assertEqual("version-0", source)
        self.assertEqual(["version-0", "version-1", "version-2"], [c[0] for c in inference.calls])
        self.assertEqual(2, len(anonymizer.calls))

    def test_exact_round_budget_with_persistent_findings(self):
        inference = RecordingInference(lambda text, attrs: (Inference("parties", "A"),))
        anonymizer = RecordingAnonymizer(lambda text, findings, hints: text + "x")
        result = adversarial_anonymize("original", ("parties",), 2,
            inference_model=inference, anonymizer_model=anonymizer)
        self.assertEqual("originalxx", result)
        self.assertEqual(2, len(inference.calls))
        self.assertEqual(2, len(anonymizer.calls))

    def test_zero_rounds_does_not_call_models_even_with_hints(self):
        inference, anonymizer, _, _ = pipeline_doubles()
        result = adversarial_anonymize("unchanged", ("parties",), 0,
            inference_model=inference, anonymizer_model=anonymizer,
            hints=(Hint("Global guidance"),))
        self.assertEqual("unchanged", result)
        self.assertFalse(inference.calls)
        self.assertFalse(anonymizer.calls)

    def test_filters_non_requested_and_null_findings(self):
        selected = Inference("parties", "A", certainty=0.8)
        inference = RecordingInference(lambda text, attrs: (
            selected, Inference("occupation", "Lawyer"), Inference("dates", None),
        ))
        anonymizer = RecordingAnonymizer(lambda text, findings, hints: "generalized")
        adversarial_anonymize("A", ("parties", "dates"), 1,
            inference_model=inference, anonymizer_model=anonymizer)
        self.assertEqual(("parties", "dates"), inference.calls[0][1])
        self.assertEqual((selected,), anonymizer.calls[0][1])

    def test_no_selected_findings_means_no_rewrite(self):
        inference = RecordingInference(lambda text, attrs: (Inference("occupation", "Lawyer"),))
        anonymizer = RecordingAnonymizer()
        self.assertEqual("text", adversarial_anonymize("text", ("parties",), 5,
            inference_model=inference, anonymizer_model=anonymizer))
        self.assertFalse(anonymizer.calls)

    def test_global_hint_forces_one_rewrite_without_local_findings(self):
        inference = RecordingInference()
        hint = Hint("Generalize a distinctive clause", ("distinctive clause",))
        anonymizer = RecordingAnonymizer(lambda text, findings, hints: "general clause")
        result = adversarial_anonymize("distinctive clause", ("parties",), 3,
            inference_model=inference, anonymizer_model=anonymizer, hints=(hint,))
        self.assertEqual("general clause", result)
        self.assertEqual(1, len(anonymizer.calls))
        self.assertEqual((hint,), anonymizer.calls[0][2])
        self.assertTrue(all("Generalize" not in call[0] for call in inference.calls))

    def test_negative_budget_rejected(self):
        inference, anonymizer, _, _ = pipeline_doubles()
        with self.assertRaises((TraceError, ValueError)):
            adversarial_anonymize("text", ("parties",), -1,
                inference_model=inference, anonymizer_model=anonymizer)

    def test_invalid_rewrite_is_rejected_by_optional_validator(self):
        inference = RecordingInference(lambda text, attrs: (Inference("parties", "A"),))
        anonymizer = RecordingAnonymizer(lambda text, findings, hints: "invented replacement")

        def reject_rewrite(before, after):
            self.assertEqual("original", before)
            self.assertEqual("invented replacement", after)
            raise TraceError("Application-specific rewrite rule")

        with self.assertRaises(TraceError):
            adversarial_anonymize("original", ("parties",), 2,
                inference_model=inference, anonymizer_model=anonymizer,
                rewrite_validator=reject_rewrite)
        self.assertEqual(1, len(anonymizer.calls))

    def test_model_response_failure_is_not_an_empty_inference(self):
        def failed_model(text, attrs):
            raise ModelResponseError("Transport did not return a complete answer")
        with self.assertRaises(ModelResponseError):
            adversarial_anonymize("text", ("parties",), 1,
                inference_model=RecordingInference(failed_model),
                anonymizer_model=RecordingAnonymizer())


class SynthesizeTraceTests(unittest.TestCase):
    def test_one_world_shared_across_segments_and_input_unchanged(self):
        original = tool_trace()
        before = copy.deepcopy(original)
        result, doubles = run_pipeline(trace=original)
        inference, anonymizer, generator, attacker = doubles
        self.assertEqual("passed_attack", result.status)
        self.assertTrue(result.succeeded)
        self.assertEqual(before, original)
        self.assertEqual(1, len(generator.calls))
        self.assertEqual(1, len(attacker.calls))
        self.assertIsNotNone(result.trace)
        self.assertEqual(len(original.segments), len(result.trace.segments))
        for segment in result.trace.segments:
            self.assertIn("Invented Buyer 1", json.dumps(segment.payload))
            self.assertNotIn("Original Buyer", json.dumps(segment.payload))
        self.assertEqual(2, result.trace.segments[1].payload["arguments"]["limit"])
        self.assertNotEqual(original.trace_id, result.trace.trace_id)
        self.assertEqual(result.trace.segments[1].call_id, result.trace.segments[2].call_id)
        self.assertIn("original", attacker.calls[0][1].lower())

    def test_ground_truth_never_enters_any_model_input(self):
        result, doubles = run_pipeline()
        self.assertTrue(result.succeeded)
        for double in doubles:
            self.assertNotIn("SCORING_ONLY_IDENTITY_SENTINEL", repr(double.calls))

    def test_reidentified_candidate_retries_with_feedback_then_passes(self):
        inference, anonymizer, generator, _ = pipeline_doubles()
        feedback = "PRIVATE_FEEDBACK_SENTINEL: distinctive payment arrangement"
        attacker = RecordingAttacker([
            AttackReport((MatterGuess(identity="Actual matter"),), feedback,
                spans=("surviving distinctive clause",), web_search_used=True),
            AttackReport((), "No original matter found", web_search_used=True),
        ])
        result, _ = run_pipeline(ground=GroundTruth(identities=("Actual matter",)), rounds=2,
            doubles=(inference, anonymizer, generator, attacker))
        self.assertTrue(result.succeeded)
        self.assertEqual(2, len(generator.calls))
        self.assertEqual(2, len(attacker.calls))
        self.assertFalse(anonymizer.calls[0][2])
        self.assertTrue(anonymizer.calls[1][2])
        self.assertIn(feedback, repr(anonymizer.calls[1][2]))
        # Every new outer pass begins with the original, not the previous invented world.
        self.assertEqual(anonymizer.calls[0][0], anonymizer.calls[1][0])
        self.assertNotIn(feedback, repr(inference.calls))
        self.assertNotIn(feedback, repr(generator.calls))
        self.assertNotIn(feedback, repr(attacker.calls))
        self.assertNotIn(feedback, repr(result))

    def test_whole_trace_hint_is_used_even_when_local_attacker_found_nothing(self):
        inference = RecordingInference()
        anonymizer = RecordingAnonymizer(lambda text, findings, hints: text.replace(
            "unusual clause", "ordinary clause") if hints else text)
        generator = RecordingGenerator(lambda profile, rules, n: SyntheticWorld({}))
        attacker = RecordingAttacker([
            AttackReport((MatterGuess(identity="Actual matter"),), "The clause identifies it",
                spans=("unusual clause",), web_search_used=True),
            AttackReport((), "No original matter found", web_search_used=True),
        ])
        original = Trace("id", (Segment("s", "user", "message", {"text": "Review unusual clause."}),))
        result, _ = run_pipeline(trace=original, ground=GroundTruth(identities=("Actual matter",)),
            rounds=2, doubles=(inference, anonymizer, generator, attacker))
        self.assertTrue(result.succeeded)
        self.assertEqual(1, len(anonymizer.calls))
        self.assertTrue(anonymizer.calls[0][2])
        self.assertIn("ordinary clause", result.trace.segments[0].payload["text"])

    def test_persistent_reidentification_fails_at_exact_budget_and_returns_no_candidate(self):
        inference, anonymizer, generator, _ = pipeline_doubles()
        attacker = RecordingAttacker([AttackReport(
            (MatterGuess(identity="Actual matter"),), "PRIVATE_REIDENTIFICATION_REASON",
            spans=("PRIVATE_SENSITIVE_SPAN",), web_search_used=True,
        )])
        result, _ = run_pipeline(ground=GroundTruth(identities=("Actual matter",)), rounds=3,
            doubles=(inference, anonymizer, generator, attacker))
        self.assertEqual("failed", result.status)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.trace)
        self.assertEqual(3, len(attacker.calls))
        self.assertEqual(3, len(generator.calls))
        self.assertNotIn("PRIVATE_REIDENTIFICATION_REASON", repr(result))
        self.assertNotIn("PRIVATE_SENSITIVE_SPAN", repr(result))
        self.assertNotIn("Actual matter", repr(result))

    def test_absent_web_search_cannot_pass(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                doubles = list(pipeline_doubles())
                doubles[3] = RecordingAttacker([AttackReport((), "No guess",
                    web_search_used=False, completed=completed)])
                result, _ = run_pipeline(doubles=doubles)
                self.assertEqual("failed", result.status)
                self.assertIsNone(result.trace)

    def test_incomplete_final_attack_cannot_pass_even_after_web_search(self):
        doubles = list(pipeline_doubles())
        doubles[3] = RecordingAttacker([AttackReport((), "Partial answer",
            web_search_used=True, completed=False)])
        result, _ = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)

    def test_same_inference_and_final_model_is_invalid_configuration(self):
        doubles = pipeline_doubles()
        doubles[3].model_id = doubles[0].model_id
        with self.assertRaises((TraceError, ValueError)):
            run_pipeline(doubles=doubles)
        self.assertFalse(doubles[0].calls)

    def test_empty_ground_truth_is_invalid_configuration(self):
        with self.assertRaises((TraceError, ValueError)):
            run_pipeline(ground=GroundTruth())

    def test_missing_world_binding_fails_before_final_attack(self):
        doubles = list(pipeline_doubles())
        doubles[2] = RecordingGenerator(lambda profile, rules, n: SyntheticWorld({}))
        result, doubles = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertFalse(doubles[3].calls)

    def test_malformed_rewrite_fails_before_world_generation(self):
        doubles = list(pipeline_doubles())
        doubles[1] = RecordingAnonymizer(lambda text, findings, hints: "not valid payload JSON")
        result, doubles = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertFalse(doubles[2].calls)
        self.assertFalse(doubles[3].calls)

    def test_shape_changed_by_rewrite_fails_closed(self):
        doubles = list(pipeline_doubles())
        doubles[1] = RecordingAnonymizer(lambda text, findings, hints: json.dumps({
            "different_key": "{{BUYER}}"
        }))
        result, doubles = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertFalse(doubles[3].calls)

    def test_final_attack_model_error_is_sanitized_and_cannot_pass(self):
        doubles = list(pipeline_doubles())
        doubles[3] = RecordingAttacker([ModelResponseError("PRIVATE_PROVIDER_ERROR_CONTENT")])
        result, _ = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertNotIn("PRIVATE_PROVIDER_ERROR_CONTENT", repr(result))

    def test_zero_outer_budget_returns_failure_without_any_model_call(self):
        result, doubles = run_pipeline(rounds=0)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertEqual("budget_exhausted", result.reason)
        for double in doubles:
            self.assertFalse(double.calls)

    def test_incoherent_generated_world_is_rejected_before_attack(self):
        doubles = list(pipeline_doubles())
        doubles[1] = RecordingAnonymizer(lambda text, findings, hints: text.replace(
            "Original Buyer", "{{BUYER}} closing {{CLOSING_DATE}} after signing {{SIGNING_DATE}}"))
        doubles[2] = RecordingGenerator(lambda profile, rules, n: SyntheticWorld({
            "BUYER": "Invented Buyer", "SIGNING_DATE": "2025-06-01", "CLOSING_DATE": "2025-01-01"
        }))
        result, doubles = run_pipeline(doubles=doubles)
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertFalse(doubles[3].calls)

    def test_final_filled_tool_arguments_must_pass_caller_validation(self):
        inference, anonymizer, generator, attacker = pipeline_doubles()

        def validate_search(arguments):
            if "Invented Buyer" in arguments["query"]:
                raise ValueError("Application rejects candidate tool arguments")

        result = synthesize_trace(tool_trace(), GroundTruth(identities=("Actual matter",)), 1, 1,
            inference_model=inference, anonymizer_model=anonymizer,
            generator=generator, final_attacker=attacker,
            tool_validators={"search": validate_search})
        self.assertEqual("failed", result.status)
        self.assertIsNone(result.trace)
        self.assertFalse(attacker.calls)


if __name__ == "__main__":
    unittest.main()
