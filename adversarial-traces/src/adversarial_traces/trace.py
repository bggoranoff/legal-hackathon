"""Trace import, structure validation and deterministic synthetic-world filling.

Only envelope identifiers are neutralized automatically. Identifying text, URLs,
document identifiers and other values inside payloads remain input to abstraction.
These helpers make no privacy claim and do not execute the recorded tool calls.
"""
from __future__ import annotations

import copy
import math
import re
import uuid
from datetime import date
from typing import Any, Callable, Mapping

from .models import AbstractProfile, Segment, SyntheticWorld, Trace, TraceError, WorldRules


_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_KEY = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_ROLES = {"system", "user", "assistant", "tool"}
_KINDS = {"message", "tool_call", "tool_result"}


def _nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TraceError(f"{label} must be a nonempty string")


def _json_value(value: Any, path: str = "payload", active: set[int] | None = None) -> None:
    """Check the strict JSON domain, including finite floats and string keys."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TraceError(f"{path} must not contain non-finite numbers")
        return
    if type(value) not in (dict, list):
        raise TraceError(f"{path} contains a non-JSON value")
    active = set() if active is None else active
    if id(value) in active:
        raise TraceError(f"{path} contains a circular reference")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TraceError(f"{path} object keys must be strings")
                _json_value(item, f"{path} object value", active)
        else:
            for item in value:
                _json_value(item, f"{path} array item", active)
    except RecursionError as exc:
        raise TraceError("JSON nesting exceeds the supported depth") from exc
    finally:
        active.remove(id(value))


def validate_shape(original: Any, rewritten: Any, *, allow_placeholders: bool) -> None:
    """Require the same object fields, array lengths and scalar JSON types.

    During abstraction an exact placeholder may replace a scalar value. A
    number must still be a number after filling; booleans are not numbers.
    Integer and floating-point values are both the JSON number type.
    """
    _json_value(original)
    _json_value(rewritten)

    def check(before: Any, after: Any) -> None:
        if (allow_placeholders and not isinstance(before, (dict, list))
                and isinstance(after, str) and _PLACEHOLDER.fullmatch(after)):
            return
        if isinstance(before, dict):
            if not isinstance(after, dict) or set(before) != set(after):
                raise TraceError("Rewriting changed object fields")
            for key in before:
                check(before[key], after[key])
        elif isinstance(before, list):
            if not isinstance(after, list) or len(before) != len(after):
                raise TraceError("Rewriting changed array structure")
            for left, right in zip(before, after):
                check(left, right)
        elif type(before) in (int, float):
            if type(after) not in (int, float):
                raise TraceError("Rewriting changed a JSON number's type")
        elif type(before) is not type(after):
            raise TraceError("Rewriting changed a scalar JSON type")

    try:
        check(original, rewritten)
    except RecursionError as exc:
        raise TraceError("JSON nesting exceeds the supported depth") from exc


def validate_trace(
    trace: Trace,
    *,
    reference: Trace | None = None,
    tool_validators: Mapping[str, Callable[[dict], None]] | None = None,
) -> None:
    """Validate trace order and tool pairs, optionally against an original.

    Tool validators receive an independent copy of a call's arguments. This
    validates syntax and supplied business rules, not live tool executability.
    """
    if not isinstance(trace, Trace):
        raise TraceError("Expected a Trace")
    _nonempty_string(trace.trace_id, "trace_id")
    if not isinstance(trace.segments, (tuple, list)) or not trace.segments:
        raise TraceError("A trace needs at least one segment")
    if tool_validators is not None and not isinstance(tool_validators, Mapping):
        raise TraceError("tool_validators must be a mapping")
    segment_ids: set[str] = set()
    call_ids: set[str] = set()
    pending: dict[str, str] = {}
    for segment in trace.segments:
        if not isinstance(segment, Segment):
            raise TraceError("Trace segments must be Segment objects")
        _nonempty_string(segment.segment_id, "segment_id")
        if segment.segment_id in segment_ids:
            raise TraceError("Duplicate segment_id")
        segment_ids.add(segment.segment_id)
        if not isinstance(segment.role, str) or segment.role not in _ROLES:
            raise TraceError("Invalid segment role")
        if not isinstance(segment.kind, str) or segment.kind not in _KINDS:
            raise TraceError("Invalid segment kind")
        if type(segment.payload) is not dict:
            raise TraceError("Segment payload must be an object")
        _json_value(segment.payload)
        if segment.kind == "message":
            if segment.role == "tool":
                raise TraceError("Tool role requires a tool result")
            if segment.tool_name is not None or segment.call_id is not None:
                raise TraceError("Messages must not have tool envelope fields")
            if set(segment.payload) != {"text"} or not isinstance(segment.payload["text"], str):
                raise TraceError("Message payload must contain only string text")
            if pending:
                raise TraceError("Message appears before pending tool results")
        else:
            _nonempty_string(segment.tool_name, "tool_name")
            _nonempty_string(segment.call_id, "call_id")
            if segment.kind == "tool_call":
                if segment.role != "assistant":
                    raise TraceError("Tool calls require the assistant role")
                if set(segment.payload) != {"arguments"} or type(segment.payload["arguments"]) is not dict:
                    raise TraceError("Tool call payload must contain only an arguments object")
                if segment.call_id in call_ids:
                    raise TraceError("Duplicate tool call_id")
                call_ids.add(segment.call_id)
                pending[segment.call_id] = segment.tool_name
                if tool_validators is not None and segment.tool_name in tool_validators:
                    validator = tool_validators[segment.tool_name]
                    if not callable(validator):
                        raise TraceError("Tool validator must be callable")
                    try:
                        validator(copy.deepcopy(segment.payload["arguments"]))
                    except Exception as exc:
                        raise TraceError("Tool argument validator rejected a call") from exc
            else:
                if segment.role != "tool" or set(segment.payload) != {"result"}:
                    raise TraceError("Tool result payload must contain only result, with tool role")
                if segment.call_id not in pending:
                    raise TraceError("Tool result has no preceding unmatched call")
                if pending.pop(segment.call_id) != segment.tool_name:
                    raise TraceError("Tool result name differs from its call")
    if pending:
        raise TraceError("Trace ends with missing tool results")
    if reference is not None:
        validate_trace(reference)
        if trace.trace_id != reference.trace_id or len(trace.segments) != len(reference.segments):
            raise TraceError("Rewriting changed trace identity or segment count")
        for original, rewritten in zip(reference.segments, trace.segments):
            envelope = lambda item: (item.segment_id, item.role, item.kind, item.tool_name, item.call_id)
            if envelope(original) != envelope(rewritten):
                raise TraceError("Rewriting changed segment envelope or order")
            validate_shape(original.payload, rewritten.payload, allow_placeholders=False)


def neutralize_trace(trace: Trace) -> Trace:
    """Replace envelope IDs and deep-copy payloads; do not scrub payload text."""
    validate_trace(trace)
    calls: dict[str, str] = {}
    segments: list[Segment] = []
    for position, segment in enumerate(trace.segments, 1):
        call_id = None
        if segment.call_id is not None:
            call_id = calls.setdefault(segment.call_id, f"call_{len(calls) + 1:04d}")
        segments.append(Segment(
            segment_id=f"segment_{position:04d}", role=segment.role, kind=segment.kind,
            payload=copy.deepcopy(segment.payload), tool_name=segment.tool_name, call_id=call_id,
        ))
    return Trace(trace_id=f"trace_{uuid.uuid4().hex}", segments=tuple(segments))


def trace_to_dict(trace: Trace) -> dict:
    """Serialize to the canonical schema without adding provenance or labels."""
    validate_trace(trace)
    return {
        "trace_id": trace.trace_id,
        "segments": [{
            "segment_id": segment.segment_id,
            "role": segment.role,
            "kind": segment.kind,
            "payload": copy.deepcopy(segment.payload),
            "tool_name": segment.tool_name,
            "call_id": segment.call_id,
        } for segment in trace.segments],
    }


SOURCE_SCHEMA_VERSION = "1.0"
_WORKFLOW = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def trace_to_source_record(trace: Trace, *, workflow: str | None = None) -> dict:
    """Serialize to the repo's ``source_traces.jsonl`` record schema.

    The output has the same top-level fields and event fields as the source
    traces, so it can be read by anything that reads them (including
    ``trace_from_dict``). IDs are fresh and carry no case name. ``workflow``
    is the generic task type (e.g. ``structure_payment``) and is optional.
    Timestamps and run metrics are not invented, so they are null.
    """
    validate_trace(trace)
    if workflow is not None and (not isinstance(workflow, str) or not _WORKFLOW.fullmatch(workflow)):
        raise TraceError("workflow must be a lowercase snake_case task type")
    case_id = f"synthetic_{uuid.uuid4().hex[:12]}"
    trace_id = f"{case_id}_t01"
    calls: dict[str, str] = {}
    events = []
    for sequence, segment in enumerate(trace.segments, 1):
        event: dict[str, Any] = {
            "event_id": f"{trace_id}.e{sequence:03d}",
            "sequence": sequence,
            "role": segment.role,
            "event_type": segment.kind,
            "timestamp": None,
            "timestamp_kind": "not recorded for synthetic trace",
        }
        if segment.kind == "message":
            event["content"] = copy.deepcopy(segment.payload["text"])
            event["model_latency_ms"] = None
            event["model_token_usage"] = None
        else:
            call_id = calls.setdefault(segment.call_id, f"{trace_id}.call{len(calls) + 1:03d}")
            event["tool_call_id"] = call_id
            event["tool_name"] = segment.tool_name
            if segment.kind == "tool_call":
                event["arguments"] = copy.deepcopy(segment.payload["arguments"])
            else:
                event["result"] = copy.deepcopy(segment.payload["result"])
                event["elapsed_ms"] = None
                event["observed_tool_run_at"] = None
        events.append(event)
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "case_id": case_id,
        "task_id": f"{case_id}_task01",
        "workflow": workflow,
        "as_of_date": None,
        "provenance": {
            "dataset_kind": "synthetic legal-agent trace",
            "author_attribution": "adversarial-traces synthesis from an abstracted source trace",
            "conversation_mode": "synthetic",
            "tool_execution_mode": "not executed; tool records are synthetic",
            "language_model_api_called_at_runtime": True,
            "timestamp_policy": "timestamps are not recorded for synthetic traces",
            "fictional_internal_dialogue": True,
            "vendor_internal_log": False,
        },
        "events": events,
        "artifacts": [],
        "final_outcome": None,
    }


def trace_from_dict(record: dict) -> Trace:
    """Import canonical records or the earlier source-trace event schema.

    Only message content, tool arguments and tool results enter the payload.
    Other record/event fields are ignored; existing IDs are always replaced.
    """
    if type(record) is not dict:
        raise TraceError("Trace record must be an object")
    if ("segments" in record) == ("events" in record):
        raise TraceError("Trace record must contain exactly one of segments or events")
    canonical = "segments" in record
    items = record["segments" if canonical else "events"]
    if not isinstance(items, (list, tuple)):
        raise TraceError("Trace segments/events must be an array")
    segments = []
    for index, item in enumerate(items, 1):
        if type(item) is not dict:
            raise TraceError("Each trace segment/event must be an object")
        try:
            if canonical:
                segment = Segment(
                    segment_id=item["segment_id"], role=item["role"], kind=item["kind"],
                    payload=copy.deepcopy(item["payload"]), tool_name=item.get("tool_name"),
                    call_id=item.get("call_id"),
                )
            else:
                if "sequence" in item and (type(item["sequence"]) is not int or item["sequence"] != index):
                    raise TraceError("Event sequence must agree with array order")
                kind = item["event_type"]
                field = {"message": "content", "tool_call": "arguments", "tool_result": "result"}.get(kind)
                if field is None:
                    raise TraceError("Unsupported event type")
                segment = Segment(
                    segment_id=item["event_id"], role=item["role"], kind=kind,
                    payload={"text" if field == "content" else field: copy.deepcopy(item[field])}, tool_name=item.get("tool_name"),
                    call_id=item.get("tool_call_id"),
                )
        except (KeyError, TypeError) as exc:
            raise TraceError("Trace record is missing a required field or has an invalid field") from exc
        segments.append(segment)
    trace = Trace(trace_id=record.get("trace_id", "import"), segments=tuple(segments))
    return neutralize_trace(trace)


def _placeholders(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_PLACEHOLDER.findall(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(_placeholders(key))
            found.update(_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_placeholders(item))
    return found


def profile_trace(trace: Trace) -> AbstractProfile:
    """Collect all payload placeholders; only the abstract trace is retained."""
    validate_trace(trace)
    names = set()
    for segment in trace.segments:
        names.update(_placeholders(segment.payload))
    return AbstractProfile(trace=copy.deepcopy(trace), placeholders=tuple(sorted(names)))


def _number(value: Any, label: str) -> int | float:
    if type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value)):
        raise TraceError(f"{label} must be a finite number")
    return value


def _date(value: Any) -> date:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise TraceError("Date bindings must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise TraceError("Date binding is not a valid calendar date") from exc


def validate_world(world: SyntheticWorld, profile: AbstractProfile, rules: WorldRules = WorldRules()) -> None:
    """Validate exact scalar bindings and explicitly configured coherence rules.

    The default checks chronological order for canonical signing, closing and
    termination keys. Other price, size, law and jurisdiction relationships
    require trusted rules from the caller; semantic plausibility is not proved.
    """
    if not isinstance(world, SyntheticWorld) or not isinstance(world.bindings, Mapping):
        raise TraceError("World must contain a bindings mapping")
    if not isinstance(profile, AbstractProfile) or not isinstance(rules, WorldRules):
        raise TraceError("Invalid abstract profile or world rules")
    actual_profile = profile_trace(profile.trace)
    if not isinstance(profile.placeholders, tuple) or profile.placeholders != actual_profile.placeholders:
        raise TraceError("Profile placeholders do not match its trace")
    bindings = world.bindings
    if any(not isinstance(key, str) or not _KEY.fullmatch(key) for key in bindings):
        raise TraceError("Binding keys must be uppercase placeholder names")
    if set(bindings) != set(profile.placeholders):
        raise TraceError("World bindings must match the expected placeholders exactly")
    for value in bindings.values():
        if type(value) not in (str, int, float, bool, type(None)):
            raise TraceError("World bindings must be JSON scalars")
        _json_value(value, "world binding")
        if isinstance(value, str) and _PLACEHOLDER.search(value):
            raise TraceError("World bindings must not contain nested placeholders")

    def bound_value(key: str) -> Any:
        if not isinstance(key, str) or key not in bindings:
            raise TraceError("World rule refers to a missing binding")
        return bindings[key]

    if not isinstance(rules.date_orders, (tuple, list)):
        raise TraceError("date_orders must be a sequence of key pairs")
    orders = list(rules.date_orders)
    for key in ("SIGNING_DATE", "CLOSING_DATE", "TERMINATION_DATE"):
        if key in bindings:
            _date(bindings[key])
    for end in ("CLOSING_DATE", "TERMINATION_DATE"):
        if "SIGNING_DATE" in bindings and end in bindings:
            orders.append(("SIGNING_DATE", end))
    for pair in orders:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TraceError("Each date order must contain two binding keys")
        if _date(bound_value(pair[0])) > _date(bound_value(pair[1])):
            raise TraceError("World dates are out of order")

    if not isinstance(rules.numeric_ranges, Mapping):
        raise TraceError("numeric_ranges must be a mapping")
    for key, bounds in rules.numeric_ranges.items():
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise TraceError("Numeric bounds must contain lower and upper limits")
        lower, upper = bounds
        if lower is not None:
            _number(lower, "Lower bound")
        if upper is not None:
            _number(upper, "Upper bound")
        if lower is not None and upper is not None and lower > upper:
            raise TraceError("Numeric lower bound exceeds upper bound")
        value = _number(bound_value(key), "Numeric binding")
        if (lower is not None and value < lower) or (upper is not None and value > upper):
            raise TraceError("World value is outside its numeric bounds")

    if not isinstance(rules.ratio_ranges, (tuple, list)):
        raise TraceError("ratio_ranges must be a sequence")
    for rule in rules.ratio_ranges:
        if not isinstance(rule, (tuple, list)) or len(rule) != 4:
            raise TraceError("Ratio rules need numerator, denominator, lower and upper")
        numerator, denominator, lower, upper = rule
        _number(lower, "Ratio lower bound")
        _number(upper, "Ratio upper bound")
        if lower > upper:
            raise TraceError("Ratio lower bound exceeds upper bound")
        top = _number(bound_value(numerator), "Ratio numerator")
        bottom = _number(bound_value(denominator), "Ratio denominator")
        if bottom == 0:
            raise TraceError("World ratio denominator must not be zero")
        try:
            ratio = top / bottom
        except OverflowError as exc:
            raise TraceError("World ratio exceeds supported numeric range") from exc
        if not math.isfinite(ratio) or not lower <= ratio <= upper:
            raise TraceError("World ratio is outside its bounds")

    if not isinstance(rules.compatible_values, (tuple, list)):
        raise TraceError("compatible_values must be a sequence")
    for rule in rules.compatible_values:
        if not isinstance(rule, (tuple, list)) or len(rule) != 3:
            raise TraceError("Compatibility rules need two keys and an allowed-values mapping")
        left_key, right_key, allowed = rule
        left, right = bound_value(left_key), bound_value(right_key)
        if not isinstance(left, str) or not isinstance(right, str) or not isinstance(allowed, Mapping):
            raise TraceError("Compatibility rules require string values and a mapping")
        if any(not isinstance(k, str) or not isinstance(v, (list, tuple, set, frozenset))
               or any(not isinstance(item, str) for item in v) for k, v in allowed.items()):
            raise TraceError("Compatibility mapping must contain collections of strings")
        if left not in allowed or right not in allowed[left]:
            raise TraceError("World contains incompatible binding values")


def fill_trace(trace: Trace, world: SyntheticWorld) -> Trace:
    """Use one binding map throughout a trace, with no model calls or retries."""
    profile = profile_trace(trace)
    validate_world(world, profile)
    bindings = world.bindings

    def inline(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def fill(value: Any, *, key: bool = False) -> Any:
        if isinstance(value, str):
            exact = _PLACEHOLDER.fullmatch(value)
            if exact and not key:
                return bindings[exact.group(1)]
            return _PLACEHOLDER.sub(lambda match: inline(bindings[match.group(1)]), value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for name, item in value.items():
                new_name = fill(name, key=True)
                if new_name in result:
                    raise TraceError("Filling placeholders produced duplicate object keys")
                result[new_name] = fill(item)
            return result
        return value

    segments = tuple(Segment(
        segment_id=segment.segment_id, role=segment.role, kind=segment.kind,
        payload=fill(segment.payload), tool_name=segment.tool_name, call_id=segment.call_id,
    ) for segment in trace.segments)
    result = Trace(trace_id=trace.trace_id, segments=segments)
    validate_trace(result)
    if profile_trace(result).placeholders:
        raise TraceError("Filled trace still contains placeholders")
    return result


def _tool_schema(required: Mapping[str, type], *, optional: Mapping[str, type] | None = None) -> Callable[[dict], None]:
    """Create small argument validators for the earlier local dataset tools."""
    optional = {} if optional is None else optional

    def check(arguments: dict) -> None:
        if not set(required).issubset(arguments) or set(arguments) - set(required) - set(optional):
            raise TraceError("Tool argument fields do not match the known tool schema")
        for name, expected in {**required, **optional}.items():
            if name not in arguments:
                continue
            if type(arguments[name]) is not expected:
                raise TraceError("Tool argument has an incorrect type")
            if expected is str and not arguments[name].strip():
                raise TraceError("Tool string argument must not be empty")
        if "limit" in arguments and arguments["limit"] < 1:
            raise TraceError("Search limit must be positive")

    return check


# Apply these to filled traces, never intermediate numeric placeholders. Callers
# can merge this mapping with custom validators, with their custom entry last.
DEFAULT_TOOL_VALIDATORS: Mapping[str, Callable[[dict], None]] = {
    "search_documents": _tool_schema({"case_id": str, "query": str}, optional={"limit": int}),
    "read_section": _tool_schema({"case_id": str, "chunk_id": str}),
    "compare_documents": _tool_schema({"case_id": str, "left_chunk_id": str, "right_chunk_id": str}),
    "draft_memo": _tool_schema({"trace_id": str, "title": str, "body": str, "artifact_directory": str}),
}
