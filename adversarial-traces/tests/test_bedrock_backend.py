"""Bedrock transport tests with a fake client; no AWS calls are made."""
import copy
import unittest

from adversarial_traces.adapters import PromptFinalAttacker, PromptInferenceModel
from adversarial_traces.bedrock_backend import BedrockConverseBackend
from adversarial_traces.models import JSONRequest, ModelResponseError, Segment, Trace, TraceError


class FakeBedrock:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


def tool_response(data, name="contract_test", stop="tool_use"):
    return {"stopReason": stop, "output": {"message": {"role": "assistant", "content": [
        {"text": "Calling the tool."},
        {"toolUse": {"toolUseId": "t1", "name": name, "input": data}},
    ]}}}


def text_response(text, stop="end_turn"):
    return {"stopReason": stop, "output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


def request(web=False):
    return JSONRequest("Return JSON", {"text": "current only"},
                       {"type": "object", "properties": {"x": {"type": "number"}},
                        "required": ["x"], "additionalProperties": False},
                       "contract_test", web_search=web)


class BedrockBackendTests(unittest.TestCase):
    def backend(self, response, **kwargs):
        self.fake = FakeBedrock(response)
        return BedrockConverseBackend("bedrock-test-model", client=self.fake, **kwargs)

    def test_tool_mode_forces_one_schema_tool_and_returns_its_input(self):
        result = self.backend(tool_response({"x": 1})).complete(request())
        self.assertEqual({"x": 1}, result.data)
        self.assertFalse(result.web_search_used)
        call = self.fake.calls[0]
        self.assertEqual("bedrock-test-model", call["modelId"])
        self.assertEqual({"tool": {"name": "contract_test"}}, call["toolConfig"]["toolChoice"])
        self.assertEqual(request().schema, call["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"])
        self.assertEqual(1, len(call["messages"]))

    def test_text_mode_parses_bare_or_fenced_json(self):
        for text in ('{"x": 2}', '```json\n{"x": 2}\n```'):
            with self.subTest(text=text):
                result = self.backend(text_response(text), structured="text").complete(request())
                self.assertEqual({"x": 2}, result.data)
                self.assertNotIn("toolConfig", self.fake.calls[0])

    def test_web_search_request_is_refused_without_calling_bedrock(self):
        backend = self.backend(tool_response({"x": 1}))
        with self.assertRaises(ModelResponseError):
            backend.complete(request(web=True))
        self.assertEqual([], self.fake.calls)

    def test_bad_responses_never_become_results(self):
        bad = [
            tool_response({"x": 1}, stop="max_tokens"),
            tool_response({"x": 1}, stop="guardrail_intervened"),
            tool_response({"x": 1}, name="other_tool"),
            text_response('{"x": 1}'),  # tool mode but no tool call
            {"stopReason": "end_turn", "output": {"message": {"content": []}}},
            RuntimeError("sensitive trace in provider error"),
        ]
        for response in bad:
            with self.subTest(response=response), self.assertRaises(ModelResponseError) as error:
                self.backend(response).complete(request())
            self.assertNotIn("sensitive trace", str(error.exception))

    def test_text_mode_rejects_invalid_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', "[]", "", "not json"):
            with self.subTest(text=text), self.assertRaises(ModelResponseError):
                self.backend(text_response(text), structured="text").complete(request())

    def test_provider_error_code_is_reported_but_not_message(self):
        class ClientError(Exception):
            response = {"Error": {"Code": "ThrottlingException", "Message": "secret input"}}
        with self.assertRaises(ModelResponseError) as error:
            self.backend(ClientError("secret input")).complete(request())
        self.assertIn("ThrottlingException", str(error.exception))
        self.assertNotIn("secret input", str(error.exception))

    def test_gpt_6_luna_defaults_to_max_and_others_to_unset(self):
        fake = FakeBedrock(tool_response({"x": 1}))
        luna = BedrockConverseBackend("global.openai.gpt-6-luna", client=fake)
        luna.complete(request())
        self.assertEqual({"reasoning": {"effort": "max"}}, fake.calls[0]["additionalModelRequestFields"])
        self.assertEqual(32000, fake.calls[0]["inferenceConfig"]["maxTokens"])
        fake = FakeBedrock(tool_response({"x": 1}))
        BedrockConverseBackend("global.openai.gpt-6-luna", client=fake, reasoning_effort="low").complete(request())
        self.assertEqual({"reasoning": {"effort": "low"}}, fake.calls[0]["additionalModelRequestFields"])
        fake = FakeBedrock(tool_response({"x": 1}))
        BedrockConverseBackend("global.openai.gpt-6-luna", client=fake, reasoning_effort=None).complete(request())
        self.assertNotIn("additionalModelRequestFields", fake.calls[0])
        fake = FakeBedrock(tool_response({"x": 1}))
        BedrockConverseBackend("global.anthropic.claude-opus-4-5-20251101-v1:0", client=fake).complete(request())
        self.assertNotIn("additionalModelRequestFields", fake.calls[0])
        self.assertEqual(8192, fake.calls[0]["inferenceConfig"]["maxTokens"])

    def test_reasoning_effort_is_sent_only_when_set(self):
        self.backend(tool_response({"x": 1})).complete(request())
        self.assertNotIn("additionalModelRequestFields", self.fake.calls[0])
        self.backend(tool_response({"x": 1}), reasoning_effort="max").complete(request())
        self.assertEqual({"reasoning": {"effort": "max"}}, self.fake.calls[0]["additionalModelRequestFields"])

    def test_config_is_validated(self):
        for kwargs in ({"model": ""}, {"model": "m", "structured": "xml"},
                       {"model": "m", "max_output_tokens": 0}, {"model": "m", "timeout": 0},
                       {"model": "m", "reasoning_effort": "banana"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TraceError):
                BedrockConverseBackend(**kwargs)

    def test_works_with_role_adapters(self):
        inference = PromptInferenceModel(self.backend(tool_response(
            {"inferences": []}, name="attribute_inferences")))
        self.assertEqual((), inference.infer("Some text", ("parties",)))
        attacker = PromptFinalAttacker(self.backend(tool_response(
            {"guesses": [], "reasoning": "Nothing found", "spans": []}, name="original_matter_attack")))
        trace = Trace("t", (Segment("s1", "user", "message", {"text": "Review the deal."}),))
        report = attacker.attack(trace, "Identify the original matter")
        self.assertEqual((), report.guesses)
        self.assertFalse(report.web_search_used)
        self.assertNotIn("must use web search", self.fake.calls[0]["system"][0]["text"])


if __name__ == "__main__":
    unittest.main()
