"""Deterministic structure tests; these do not test model privacy efficacy."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import unittest

from adversarial_traces.models import AbstractProfile, Segment, SyntheticWorld, Trace, TraceError, WorldRules
from adversarial_traces.trace import (
    trace_to_source_record,
    DEFAULT_TOOL_VALIDATORS, fill_trace, neutralize_trace, profile_trace,
    trace_from_dict, trace_to_dict, validate_shape, validate_trace, validate_world,
)


def message(text="Buyer buys Target", *, trace_id="secret-deal"):
    return Trace(trace_id, (Segment("secret-segment", "user", "message", {"text": text}),))


def tool_trace(arguments=None, result=None):
    return Trace("secret-deal", (
        Segment("request", "user", "message", {"text": "Review the purchase"}),
        Segment("call", "assistant", "tool_call", {"arguments": arguments or {"case_id": "secret-deal", "chunk_id": "secret-document"}}, "read_section", "secret-call"),
        Segment("result", "tool", "tool_result", {"result": result if result is not None else {"text": "Purchase terms"}}, "read_section", "secret-call"),
        Segment("answer", "assistant", "message", {"text": "Here is the review"}),
    ))


def world_profile(bindings):
    return profile_trace(message(" ".join("{{" + key + "}}" for key in bindings)))


class TraceValidationTests(unittest.TestCase):
    def test_valid_messages_and_tools(self):
        validate_trace(message())
        validate_trace(tool_trace(), tool_validators=DEFAULT_TOOL_VALIDATORS)

    def test_invalid_input_is_trace_error(self):
        cases = [None, Trace("", ()), Trace("t", ()), Trace("t", (None,)),
                 Trace("t", (Segment("s", [], "message", {"text": "x"}),)),
                 Trace("t", (Segment("s", "tool", "message", {"text": "x"}),)),
                 Trace("t", (Segment("s", "user", "message", {"content": "x"}),)),
                 Trace("t", (Segment("s", "user", "message", {"text": 3}),))]
        for trace in cases:
            with self.subTest(trace=trace), self.assertRaises(TraceError):
                validate_trace(trace)

    def test_json_domain_and_cycles(self):
        for value in [float("nan"), float("inf"), -float("inf"), (1, 2), {3: "x"}, {1, 2}]:
            with self.subTest(value=value), self.assertRaises(TraceError):
                validate_trace(tool_trace(result=value))
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(TraceError):
            validate_trace(tool_trace(result=cycle))

    def test_duplicate_ids_and_unpaired_tools(self):
        original = tool_trace()
        variants = [
            replace(original, segments=original.segments + (original.segments[-1],)),
            replace(original, segments=original.segments[:2]),
            replace(original, segments=(original.segments[2],)),
            replace(original, segments=(original.segments[0], original.segments[1], original.segments[3], original.segments[2])),
            replace(original, segments=(original.segments[0], original.segments[1], replace(original.segments[2], tool_name="other"), original.segments[3])),
        ]
        for trace in variants:
            with self.subTest(trace=trace), self.assertRaises(TraceError):
                validate_trace(trace)

    def test_parallel_calls_allowed(self):
        a, b, c, d = tool_trace().segments
        trace = Trace("t", (a, b, replace(b, segment_id="call2", call_id="call2"),
                            replace(c, segment_id="result2", call_id="call2"), c, d))
        validate_trace(trace)

    def test_reference_fixes_envelope_and_payload_shape(self):
        original = tool_trace()
        changed = replace(original, segments=tuple(replace(s, payload={"text": "New answer"}) if s.kind == "message" else s for s in original.segments))
        validate_trace(changed, reference=original)
        with self.assertRaises(TraceError):
            validate_trace(replace(changed, trace_id="different"), reference=original)
        with self.assertRaises(TraceError):
            validate_trace(replace(changed, segments=tuple(replace(s, tool_name="other") if s.tool_name else s for s in changed.segments)), reference=original)

    def test_custom_validator_gets_copy_and_errors_are_wrapped(self):
        trace = tool_trace()
        def mutating(arguments):
            arguments.clear()
        validate_trace(trace, tool_validators={"read_section": mutating})
        self.assertIn("case_id", trace.segments[1].payload["arguments"])
        def rejection(arguments):
            raise ValueError("invalid")
        with self.assertRaises(TraceError):
            validate_trace(trace, tool_validators={"read_section": rejection})

    def test_known_tool_argument_schema(self):
        valid = tool_trace()
        with self.assertRaises(TraceError):
            validate_trace(tool_trace(arguments={"case_id": "x", "chunk_id": 17}), tool_validators=DEFAULT_TOOL_VALIDATORS)
        with self.assertRaises(TraceError):
            validate_trace(tool_trace(arguments={"case_id": "x", "chunk_id": "c", "surprise": True}), tool_validators=DEFAULT_TOOL_VALIDATORS)
        validate_trace(valid, tool_validators=DEFAULT_TOOL_VALIDATORS)


class SerializationTests(unittest.TestCase):
    def test_neutralization_fresh_ids_and_no_input_mutation(self):
        original = tool_trace()
        before = copy.deepcopy(original)
        first, second = neutralize_trace(original), neutralize_trace(original)
        self.assertNotEqual(first.trace_id, second.trace_id)
        self.assertNotIn("secret", first.trace_id)
        self.assertEqual(first.segments[1].call_id, first.segments[2].call_id)
        self.assertEqual(first.segments[0].segment_id, "segment_0001")
        self.assertEqual(first.segments[1].payload["arguments"]["case_id"], "secret-deal")
        first.segments[1].payload["arguments"]["case_id"] = "edited"
        self.assertEqual(original, before)

    def test_canonical_import_roundtrip_preserves_payloads_not_ids(self):
        original = tool_trace()
        serialized = trace_to_dict(original)
        imported = trace_from_dict(serialized)
        self.assertNotEqual(imported.trace_id, original.trace_id)
        self.assertEqual([s.payload for s in original.segments], [s.payload for s in imported.segments])
        serialized["segments"][0]["payload"]["text"] = "changed"
        self.assertNotEqual(original.segments[0].payload["text"], "changed")

    def test_legacy_import_drops_envelope_metadata_not_raw_tool_payloads(self):
        record = {
            "trace_id": "identifying-trace", "case_id": "private-case", "provenance": {"secret": "top-metadata"},
            "events": [
                {"sequence": 1, "event_id": "original-1", "role": "user", "event_type": "message", "content": "Review Buyer", "timestamp": "sensitive-time"},
                {"sequence": 2, "event_id": "original-2", "role": "assistant", "event_type": "tool_call", "tool_name": "fetch", "tool_call_id": "private-call", "arguments": {"url": "https://example.test/Buyer"}},
                {"sequence": 3, "event_id": "original-3", "role": "tool", "event_type": "tool_result", "tool_name": "fetch", "tool_call_id": "private-call", "result": {"text": "Buyer", "document_id": "private-document"}, "artifact_path": "hidden-artifact"},
            ],
        }
        before = copy.deepcopy(record)
        trace = trace_from_dict(record)
        serialized = json.dumps(trace_to_dict(trace))
        for discarded in ("identifying-trace", "top-metadata", "sensitive-time", "hidden-artifact", "original-1", "private-call"):
            self.assertNotIn(discarded, serialized)
        for retained in ("https://example.test/Buyer", "private-document", "Buyer"):
            self.assertIn(retained, serialized)
        self.assertEqual(trace.segments[0].payload, {"text": "Review Buyer"})
        self.assertEqual(record, before)

    def test_malformed_imports_raise_deliberately(self):
        for record in [None, {}, {"segments": [], "events": []}, {"segments": "bad"}, {"segments": [None]}, {"events": [{"event_type": "unknown"}]}]:
            with self.subTest(record=record), self.assertRaises(TraceError):
                trace_from_dict(record)


class SourceRecordTests(unittest.TestCase):
    def trace(self):
        return Trace("t", (
            Segment("a", "user", "message", {"text": "Summarize the deal."}),
            Segment("b", "assistant", "tool_call", {"arguments": {"case_id": "c", "query": "q"}},
                    tool_name="search_documents", call_id="x"),
            Segment("c", "tool", "tool_result", {"result": {"matched_chunks": 0, "results": []}},
                    tool_name="search_documents", call_id="x"),
            Segment("d", "assistant", "message", {"text": "Done."}),
        ))

    def test_round_trips_through_the_source_importer(self):
        record = trace_to_source_record(self.trace(), workflow="structure_payment")
        back = trace_from_dict(json.loads(json.dumps(record)))
        self.assertEqual([seg.payload for seg in self.trace().segments], [seg.payload for seg in back.segments])
        self.assertEqual("structure_payment", record["workflow"])
        self.assertEqual(list(range(1, 5)), [e["sequence"] for e in record["events"]])
        self.assertEqual(record["events"][1]["tool_call_id"], record["events"][2]["tool_call_id"])
        self.assertTrue(record["trace_id"].startswith(record["case_id"]))

    def test_ids_are_fresh_and_workflow_is_checked(self):
        first = trace_to_source_record(self.trace())
        self.assertNotEqual(first["case_id"], trace_to_source_record(self.trace())["case_id"])
        self.assertIsNone(first["workflow"])
        for bad in ("Microsoft Activision", "", 3):
            with self.subTest(bad=bad), self.assertRaises(TraceError):
                trace_to_source_record(self.trace(), workflow=bad)


class ShapeAndFillingTests(unittest.TestCase):
    def test_scalar_placeholders_allow_abstraction_but_not_type_errors(self):
        validate_shape({"amount": 5, "flag": True}, {"amount": "{{PRICE}}", "flag": "{{FLAG}}"}, allow_placeholders=True)
        validate_shape({"amount": 5}, {"amount": 5.5}, allow_placeholders=False)
        for after in ({"amount": "{{PRICE}}"}, {"amount": True}, {"amount": "about 5"}, {"other": 5}):
            with self.subTest(after=after), self.assertRaises(TraceError):
                validate_shape({"amount": 5}, after, allow_placeholders=False)
        with self.assertRaises(TraceError):
            validate_shape({"a": [1]}, {"a": [1, 2]}, allow_placeholders=True)
        with self.assertRaises(TraceError):
            validate_shape({"a": {}}, {"a": "{{OBJECT}}"}, allow_placeholders=True)

    def test_profile_collects_nested_values_and_keys(self):
        trace = tool_trace(result={"{{BUYER}}": ["{{PRICE}}", "Paid {{PRICE}} to {{TARGET}}"]})
        profile = profile_trace(trace)
        self.assertEqual(profile.placeholders, ("BUYER", "PRICE", "TARGET"))
        trace.segments[2].payload["result"].clear()
        self.assertTrue(profile.trace.segments[2].payload["result"])

    def test_one_world_reused_and_scalars_keep_types(self):
        trace = tool_trace(arguments={"query": "Find {{BUYER}}", "limit": "{{LIMIT}}"}, result={"{{BUYER}}": ["{{LIMIT}}", "{{FLAG}}", "{{NONE}}", "{{PRICE}}", "{{BUYER}} paid {{PRICE}}"]})
        world = SyntheticWorld({"BUYER": "Invented Cedar", "LIMIT": 5, "FLAG": False, "NONE": None, "PRICE": 7.5})
        filled = fill_trace(trace, world)
        self.assertEqual(filled.segments[1].payload["arguments"], {"query": "Find Invented Cedar", "limit": 5})
        self.assertEqual(filled.segments[2].payload["result"], {"Invented Cedar": [5, False, None, 7.5, "Invented Cedar paid 7.5"]})
        self.assertEqual(filled.trace_id, trace.trace_id)
        self.assertEqual(profile_trace(filled).placeholders, ())
        self.assertEqual(trace.segments[1].payload["arguments"]["limit"], "{{LIMIT}}")

    def test_inline_scalar_rendering(self):
        filled = fill_trace(message("{{FLAG}}/{{NONE}}/{{VALUE}}"), SyntheticWorld({"FLAG": False, "NONE": None, "VALUE": 3}))
        self.assertEqual(filled.segments[0].payload["text"], "false/null/3")

    def test_missing_extra_nested_and_colliding_bindings_rejected(self):
        trace = tool_trace(result={"{{BUYER}}": 1, "Cedar": 2})
        for world in (SyntheticWorld({}), SyntheticWorld({"BUYER": "Elm", "EXTRA": 1}), SyntheticWorld({"BUYER": "{{OTHER}}"}), SyntheticWorld({"BUYER": "Cedar"})):
            with self.subTest(world=world), self.assertRaises(TraceError):
                fill_trace(trace, world)

    def test_dynamic_key_filling_supported_but_not_shape_preservation(self):
        trace = tool_trace(result={"{{PARTY}}": 1})
        filled = fill_trace(trace, SyntheticWorld({"PARTY": "Cedar"}))
        with self.assertRaises(TraceError):
            validate_shape(trace.segments[2].payload, filled.segments[2].payload, allow_placeholders=False)


class WorldValidationTests(unittest.TestCase):
    def test_default_dates_and_unrelated_closing_termination_order(self):
        bindings = {"SIGNING_DATE": "2025-01-01", "CLOSING_DATE": "2025-09-01", "TERMINATION_DATE": "2025-04-01"}
        profile = world_profile(bindings)
        validate_world(SyntheticWorld(bindings), profile)
        for key, value in [("SIGNING_DATE", "2025-10-01"), ("CLOSING_DATE", "2025-02-30"), ("SIGNING_DATE", "20250101")]:
            with self.subTest(key=key, value=value), self.assertRaises(TraceError):
                validate_world(SyntheticWorld({**bindings, key: value}), profile)

    def test_trusted_price_size_law_and_date_rules(self):
        bindings = {"PRICE": 100, "REVENUE": 50, "LAW": "Delaware", "JURISDICTION": "US", "START": "2026-01-01", "END": "2026-04-01"}
        rules = WorldRules(
            date_orders=(("START", "END"),), numeric_ranges={"PRICE": (1, 500)},
            ratio_ranges=(("PRICE", "REVENUE", 1, 4),),
            compatible_values=(("JURISDICTION", "LAW", {"US": ["Delaware", "New York"]}),),
        )
        profile = world_profile(bindings)
        validate_world(SyntheticWorld(bindings), profile, rules)
        for override in ({"PRICE": 0}, {"REVENUE": 0}, {"REVENUE": 1}, {"LAW": "England"}, {"START": "2027-01-01"}, {"PRICE": True}):
            with self.subTest(override=override), self.assertRaises(TraceError):
                validate_world(SyntheticWorld({**bindings, **override}), profile, rules)

    def test_bad_rules_and_bindings_raise_trace_error(self):
        profile = world_profile({"VALUE": 1})
        for bindings in ({"VALUE": {}}, {"VALUE": float("nan")}, {"VALUE": "{{OTHER}}"}, {"lower": 1}):
            with self.subTest(bindings=bindings), self.assertRaises(TraceError):
                validate_world(SyntheticWorld(bindings), profile)
        for rules in (WorldRules(date_orders=("bad",)), WorldRules(numeric_ranges={"VALUE": (3, 2)}), WorldRules(numeric_ranges={"VALUE": (False, 3)}), WorldRules(ratio_ranges=(("VALUE", "MISSING", 1, 2),)), WorldRules(compatible_values=(("VALUE", "VALUE", {"x": "bad"}),))):
            with self.subTest(rules=rules), self.assertRaises(TraceError):
                validate_world(SyntheticWorld({"VALUE": 1}), profile, rules)
        with self.assertRaises(TraceError):
            validate_world(SyntheticWorld({"VALUE": 1}), AbstractProfile(profile.trace, ("OTHER",)))


class OptionalDatasetIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ADVERSARIAL_TRACES_DATASET"), "Set ADVERSARIAL_TRACES_DATASET to test the earlier source JSONL")
    def test_prior_dataset_imports(self):
        path = Path(os.environ["ADVERSARIAL_TRACES_DATASET"])
        count = 0
        ids = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trace = trace_from_dict(json.loads(line))
                validate_trace(trace, tool_validators=DEFAULT_TOOL_VALIDATORS)
                self.assertNotIn(trace.trace_id, ids)
                ids.add(trace.trace_id)
                count += 1
        self.assertGreater(count, 0)




class SourceFormatDatasetTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ADVERSARIAL_TRACES_DATASET"), "Set ADVERSARIAL_TRACES_DATASET to compare with the source JSONL")
    def test_output_has_the_same_fields_as_every_source_trace(self):
        path = Path(os.environ["ADVERSARIAL_TRACES_DATASET"])
        for line in path.read_text(encoding="utf-8").splitlines():
            source = json.loads(line)
            record = trace_to_source_record(trace_from_dict(source), workflow=source["workflow"])
            self.assertEqual(set(source), set(record))
            for before, after in zip(source["events"], record["events"]):
                self.assertEqual(set(before), set(after))
                for field in ("sequence", "role", "event_type", "tool_name", "content", "arguments", "result"):
                    self.assertEqual(before.get(field), after.get(field))


if __name__ == "__main__":
    unittest.main()
