"""Offline contract tests; these do not measure model anonymization quality."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

from adversarial_traces.adapters import (
    PromptAnonymizerModel, PromptFinalAttacker, PromptInferenceModel,
    PromptWorldGenerator,
)
from adversarial_traces.models import (
    Hint, Inference, JSONRequest, JSONResponse, ModelResponseError,
    Segment, Trace, TraceError, WorldRules,
)
from adversarial_traces.openai_backend import OpenAIResponsesBackend
from adversarial_traces.trace import profile_trace


class RecordingBackend:
    model_id = "offline-contract-double"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def sample_trace(text="Invented Cedar buys Invented Elm."):
    return Trace("trace_1", (Segment("segment_1", "user", "message", {"text": text}),))


def attack_data():
    return {"guesses": [], "reasoning": "No supported match from the search.", "spans": []}


def inference_data():
    return {"inferences": [{"attribute": "parties", "value": "Elm", "certainty": 0.8,
                            "reasoning": "The buyer is named.", "spans": ["Elm"]}]}


class AdapterTests(unittest.TestCase):
    def test_inference_is_current_text_only_and_fresh(self):
        backend = RecordingBackend(JSONResponse(inference_data()), JSONResponse({"inferences": []}))
        adapter = PromptInferenceModel(backend)
        guesses = adapter.infer("Elm buys a company.", ("parties",))
        self.assertEqual(guesses[0].value, "Elm")
        self.assertEqual(adapter.infer("A buyer acquires a business.", ("parties",)), ())
        first, second = backend.requests
        self.assertIsNot(first, second)
        self.assertEqual(second.payload, {"text": "A buyer acquires a business.", "attributes": ["parties"]})
        self.assertNotIn("Elm", json.dumps(second.payload))
        self.assertFalse(first.web_search)

    def test_inference_malformed_is_not_empty_success(self):
        malformed = [
            {}, {"inferences": "none"},
            {"inferences": [{**inference_data()["inferences"][0], "certainty": True}]},
            {"inferences": [{**inference_data()["inferences"][0], "certainty": float("nan")}]},
            {"inferences": [{**inference_data()["inferences"][0], "attribute": "occupation"}]},
        ]
        for data in malformed:
            with self.subTest(data=data), self.assertRaises(ModelResponseError):
                PromptInferenceModel(RecordingBackend(JSONResponse(data))).infer("Elm", ("parties",))

    def test_backend_error_does_not_expose_trace(self):
        backend = RecordingBackend(RuntimeError("secret customer text"))
        with self.assertRaises(ModelResponseError) as error:
            PromptInferenceModel(backend).infer("Elm", ("parties",))
        self.assertNotIn("secret", str(error.exception))

    def test_hint_is_separate_even_without_local_inferences(self):
        backend = RecordingBackend(JSONResponse({"text": "An acquisition."}))
        text = "A peculiar three-stage acquisition."
        hint = Hint("Generalize the distinctive stages.", ("three-stage",), "segment_1")
        result = PromptAnonymizerModel(backend).anonymize(text, (), hints=(hint,))
        self.assertEqual(result, "An acquisition.")
        request = backend.requests[0]
        self.assertEqual(request.payload["text"], text)
        self.assertEqual(request.payload["inferences"], [])
        self.assertEqual(request.payload["hints"][0]["reasoning"], hint.reasoning)
        self.assertIn("NEVER invent", request.system)
        self.assertNotIn("json", request.payload)

    def test_json_content_is_sent_and_returned_as_an_object(self):
        # Quotes inside values must survive without the model escaping them.
        backend = RecordingBackend(JSONResponse({"json": {"text": 'The "{{BUYER}}" deal'}}))
        text = json.dumps({"text": 'The "Elm" deal'})
        result = PromptAnonymizerModel(backend).anonymize(text, ())
        self.assertEqual({"text": 'The "{{BUYER}}" deal'}, json.loads(result))
        request = backend.requests[0]
        self.assertEqual({"text": 'The "Elm" deal'}, request.payload["json"])
        self.assertNotIn("text", request.payload)
        self.assertIn("preserve all object keys", request.system)
        with self.assertRaises(ModelResponseError):
            PromptAnonymizerModel(RecordingBackend(JSONResponse({"text": "x"}))).anonymize(text, ())

    def test_anonymizer_rejects_nontext_and_drops_echoed_fields(self):
        for data in ({"text": 4}, {"identity": "leak"}):
            with self.subTest(data=data), self.assertRaises(ModelResponseError):
                PromptAnonymizerModel(RecordingBackend(JSONResponse(data))).anonymize("Elm", ())
        echoed = {"text": "ok", "inferences": [], "hints": [], "identity": "unused"}
        self.assertEqual("ok", PromptAnonymizerModel(RecordingBackend(JSONResponse(echoed))).anonymize("Elm", ()))

    def test_json_rewrite_repairs_echoed_and_dropped_null_fields_only(self):
        original = {"result": {"items": [{"id": "Elm-1", "date": None}], "note": "Elm"}}
        reply = {"json": {"result": {"items": [{"id": "{{DOC}}"}], "note": "{{BUYER}}"},
                          "hints": [], "inferences": []}}
        out = PromptAnonymizerModel(RecordingBackend(JSONResponse(reply))).anonymize(json.dumps(original), ())
        self.assertEqual({"result": {"items": [{"id": "{{DOC}}", "date": None}], "note": "{{BUYER}}"}},
                         json.loads(out))
        # A dropped field that had a real value is not restored.
        reply = {"json": {"result": {"items": [{"date": None}], "note": "{{BUYER}}"}}}
        out = PromptAnonymizerModel(RecordingBackend(JSONResponse(reply))).anonymize(json.dumps(original), ())
        self.assertNotIn("Elm", out)

    def test_findings_that_only_point_at_placeholders_are_dropped(self):
        item = inference_data()["inferences"][0]
        data = {"inferences": [
            {**item, "value": "{{BUYER}}; {{TARGET}}", "spans": []},
            {**item, "value": "Buyer and target", "spans": ["{{BUYER}}_{{TARGET}}_{{YEAR}}"]},
            {**item, "value": "Elm", "spans": ["Elm"]},
        ]}
        text = "Elm {{BUYER}}_{{TARGET}}_{{YEAR}}"
        result = PromptInferenceModel(RecordingBackend(JSONResponse(data))).infer(text, ("parties",))
        self.assertEqual(["Elm"], [inference.value for inference in result])

    def test_generator_gets_abstract_profile_and_rules_only(self):
        profile = profile_trace(sample_trace("{{BUYER}} buys {{TARGET}} for {{PRICE}}."))
        backend = RecordingBackend(JSONResponse({"bindings": [
            {"key": "BUYER", "value": "Invented Cedar"},
            {"key": "TARGET", "value": "Invented Elm"},
            {"key": "PRICE", "value": 100},
        ]}))
        world = PromptWorldGenerator(backend).generate(profile, WorldRules(numeric_ranges={"PRICE": (1, 1000)}))
        self.assertEqual(world.bindings["PRICE"], 100)
        self.assertEqual(set(backend.requests[0].payload), {"profile", "rules"})
        self.assertEqual(backend.requests[0].payload["profile"]["placeholders"], ["BUYER", "PRICE", "TARGET"])

    def test_generator_rejects_duplicate_or_nonscalar_bindings(self):
        profile = profile_trace(sample_trace("{{BUYER}}"))
        cases = [[{"key": "BUYER", "value": []}],
                 [{"key": "BUYER", "value": "A"}, {"key": "BUYER", "value": "B"}]]
        for bindings in cases:
            with self.subTest(bindings=bindings), self.assertRaises(ModelResponseError):
                PromptWorldGenerator(RecordingBackend(JSONResponse({"bindings": bindings}))).generate(profile, WorldRules())

    def test_generator_rejects_missing_extra_or_nested_bindings_as_candidates(self):
        profile = profile_trace(sample_trace("{{BUYER}}"))
        cases = [[], [{"key": "BUYER", "value": "{{TARGET}}"}],
                 [{"key": "BUYER", "value": "A"}, {"key": "TARGET", "value": "B"}]]
        for bindings in cases:
            with self.subTest(bindings=bindings), self.assertRaises(TraceError):
                PromptWorldGenerator(RecordingBackend(JSONResponse({"bindings": bindings}))).generate(profile, WorldRules())

    def test_world_coherence_failure_is_retryable_candidate_failure(self):
        profile = profile_trace(sample_trace("Signed {{SIGNING_DATE}}. Closed {{CLOSING_DATE}}."))
        backend = RecordingBackend(JSONResponse({"bindings": [
            {"key": "SIGNING_DATE", "value": "2024-05-01"},
            {"key": "CLOSING_DATE", "value": "2024-04-01"},
        ]}))
        with self.assertRaises(TraceError):
            PromptWorldGenerator(backend).generate(profile, WorldRules())

    def test_final_attacker_requires_transport_search_when_enabled(self):
        for claimed in (attack_data(), {**attack_data(), "web_search_used": True}):
            backend = RecordingBackend(JSONResponse(claimed, web_search_used=False))
            with self.assertRaises(ModelResponseError):
                PromptFinalAttacker(backend, web_search=True).attack(sample_trace(), "Identify the original matter")

    def test_final_attacker_search_off_by_default(self):
        backend = RecordingBackend(JSONResponse(attack_data(), web_search_used=False))
        report = PromptFinalAttacker(backend).attack(sample_trace(), "Identify the original matter")
        self.assertFalse(report.web_search_used)
        self.assertTrue(report.completed)
        self.assertFalse(backend.requests[0].web_search)
        self.assertNotIn("must use web search", backend.requests[0].system)

    def test_final_attacker_accepts_verified_search_and_no_ground(self):
        backend = RecordingBackend(JSONResponse(attack_data(), web_search_used=True))
        report = PromptFinalAttacker(backend, web_search=True).attack(sample_trace(), "Identify the original matter")
        self.assertTrue(report.web_search_used)
        self.assertTrue(report.completed)
        self.assertEqual(report.guesses, ())
        self.assertEqual(set(backend.requests[0].payload), {"trace", "instruction"})
        self.assertTrue(backend.requests[0].web_search)

    def test_list_value_is_joined_into_one_guess(self):
        data = {"inferences": [{**inference_data()["inferences"][0], "value": ["Elm", "Oak"]}]}
        result = PromptInferenceModel(RecordingBackend(JSONResponse(data))).infer("Elm", ("parties",))
        self.assertEqual("Elm; Oak", result[0].value)
        labelled = {"inferences": [{**inference_data()["inferences"][0], "value": {"signing": "2022", "closing": None}}]}
        result = PromptInferenceModel(RecordingBackend(JSONResponse(labelled))).infer("Elm", ("parties",))
        self.assertEqual("signing: 2022", result[0].value)
        nested = {"inferences": [{**inference_data()["inferences"][0], "value": [["Elm"]]}]}
        with self.assertRaises(ModelResponseError):
            PromptInferenceModel(RecordingBackend(JSONResponse(nested))).infer("Elm", ("parties",))

    def test_misquoted_evidence_is_dropped_not_fatal(self):
        data = {"inferences": [{**inference_data()["inferences"][0], "spans": ["Elm", "invented evidence"]}]}
        result = PromptInferenceModel(RecordingBackend(JSONResponse(data))).infer("Elm", ("parties",))
        self.assertEqual(("Elm",), result[0].spans)
        report = PromptFinalAttacker(RecordingBackend(JSONResponse(
            {**attack_data(), "spans": ["not present in trace"]}))).attack(sample_trace(), "Identify")
        self.assertEqual((), report.spans)

    def test_final_attacker_rejects_malformed_guess(self):
        cases = [
            {**attack_data(), "guesses": [{"identity": None, "parties": [], "certainty": 0.1}]},
            {**attack_data(), "web_search_used": True},
        ]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ModelResponseError):
                PromptFinalAttacker(RecordingBackend(JSONResponse(data, True))).attack(sample_trace(), "Identify")


def response_record(text='{"inferences": []}', *, search=False):
    output = []
    if search:
        output.append({"type": "web_search_call", "status": "completed",
                       "action": {"type": "search", "queries": ["acquisition terms"]}})
    output.append({"type": "message", "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": text}]})
    return {"status": "completed", "error": None, "incomplete_details": None, "output": output}


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


class OpenAIBackendTests(unittest.TestCase):
    def request(self, web=False):
        return JSONRequest("Return JSON", {"text": "current only"},
                           {"type": "object", "properties": {}, "additionalProperties": False},
                           "contract_test", web_search=web)

    def backend(self, response):
        self.fake = FakeResponses(response)
        return OpenAIResponsesBackend("configured-test-model", client=SimpleNamespace(responses=self.fake))

    def test_fresh_response_requests_no_storage_or_history(self):
        backend = self.backend(response_record())
        backend.complete(self.request())
        backend.complete(self.request())
        self.assertEqual(len(self.fake.calls), 2)
        for call in self.fake.calls:
            self.assertEqual(call["model"], "configured-test-model")
            self.assertFalse(call["store"])
            self.assertNotIn("previous_response_id", call)
            self.assertNotIn("conversation", call)
            self.assertNotIn("tools", call)
            self.assertEqual(len(call["input"]), 2)
            self.assertTrue(call["text"]["format"]["strict"])

    def test_search_is_required_and_verified_from_output(self):
        backend = self.backend(response_record(search=True))
        result = backend.complete(self.request(web=True))
        self.assertTrue(result.web_search_used)
        self.assertEqual(self.fake.calls[0]["tool_choice"], "required")
        self.assertEqual(self.fake.calls[0]["tools"], [{"type": "web_search"}])

    def test_no_completed_search_never_passes(self):
        missing = response_record()
        unfinished = response_record(search=True)
        unfinished["output"][0]["status"] = "failed"
        page_only = response_record(search=True)
        page_only["output"][0]["action"] = {"type": "open_page", "url": "https://example.com"}
        for record in (missing, unfinished, page_only):
            with self.subTest(record=record), self.assertRaises(ModelResponseError):
                self.backend(record).complete(self.request(web=True))

    def test_truncation_refusal_or_api_error_never_becomes_empty_result(self):
        records = []
        for status in ("incomplete", "failed", "cancelled", "in_progress"):
            record = response_record()
            record["status"] = status
            records.append(record)
        refusal = response_record()
        refusal["output"][0]["content"] = [{"type": "refusal", "refusal": "Declined"}]
        records.append(refusal)
        records.append(RuntimeError("sensitive trace in provider error"))
        for record in records:
            with self.subTest(record=record), self.assertRaises(ModelResponseError) as error:
                self.backend(record).complete(self.request())
            self.assertNotIn("sensitive trace", str(error.exception))

    def test_invalid_json_duplicate_keys_and_nonfinite_are_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '[]', '', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(ModelResponseError):
                self.backend(response_record(text)).complete(self.request())

    def test_sdk_object_response_shape(self):
        record = response_record(search=True)
        def objects(value):
            if isinstance(value, dict):
                return SimpleNamespace(**{key: objects(item) for key, item in value.items()})
            if isinstance(value, list):
                return [objects(item) for item in value]
            return value
        result = self.backend(objects(record)).complete(self.request(web=True))
        self.assertTrue(result.web_search_used)

    def test_explicit_model_and_bounded_transport_config(self):
        for kwargs in ({"model": ""}, {"model": "m", "max_output_tokens": True},
                       {"model": "m", "timeout": float("inf")}, {"model": "m", "timeout": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TraceError):
                OpenAIResponsesBackend(**kwargs)


if __name__ == "__main__":
    unittest.main()
