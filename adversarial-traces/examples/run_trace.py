"""Explicit, opt-in API runner for one selected JSONL trace.

Ground aliases are private scoring inputs, never a model argument. Only a
candidate that passes the configured attack is written. This does not establish
anonymity or differential privacy. Existing outputs require --overwrite.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from adversarial_traces import (
    DEFAULT_TOOL_VALIDATORS, GroundTruth, WorldRules,
    synthesize_trace, trace_from_dict, trace_to_dict,
)
from adversarial_traces.adapters import (
    PromptAnonymizerModel, PromptFinalAttacker, PromptInferenceModel, PromptWorldGenerator,
)
from adversarial_traces.matching import validate_ground
from adversarial_traces.openai_backend import OpenAIResponsesBackend


def _json(text):
    def reject_constant(value):
        raise ValueError("Non-finite JSON")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)

    def finite(item):
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Non-finite JSON number")
        if type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)

    finite(value)
    return value


def _ground(path):
    value = _json(path.read_text(encoding="utf-8"))
    if type(value) is not dict or set(value) - {"identities", "party_aliases"}:
        raise ValueError("Invalid ground fields")
    ground = GroundTruth(identities=value.get("identities", ()),
                         party_aliases=value.get("party_aliases", ()))
    validate_ground(ground)
    return ground


def _rules(path):
    if path is None:
        return WorldRules()
    value = _json(path.read_text(encoding="utf-8"))
    if type(value) is not dict or set(value) - {field.name for field in fields(WorldRules)}:
        raise ValueError("Invalid rule fields")

    def key(item):
        if type(item) is not str or not re.fullmatch(r"[A-Z][A-Z0-9_]*", item):
            raise ValueError("Invalid rule binding key")
        return item

    def sequence(item, length):
        if type(item) is not list or len(item) != length:
            raise ValueError("Invalid rule length")
        return item

    def number(item, nullable=False):
        if item is None and nullable:
            return item
        if type(item) not in (int, float) or (type(item) is float and not math.isfinite(item)):
            raise ValueError("Invalid numeric rule")
        return item

    dates = value.get("date_orders", [])
    ratios = value.get("ratio_ranges", [])
    compatible = value.get("compatible_values", [])
    numeric = value.get("numeric_ranges", {})
    if any(type(items) is not list for items in (dates, ratios, compatible)) or type(numeric) is not dict:
        raise ValueError("Invalid rule collections")
    for pair in dates:
        for item in sequence(pair, 2):
            key(item)
    for name, bounds in numeric.items():
        key(name)
        low, high = sequence(bounds, 2)
        number(low, nullable=True)
        number(high, nullable=True)
        if low is not None and high is not None and low > high:
            raise ValueError("Invalid numeric interval")
    for rule in ratios:
        numerator, denominator, low, high = sequence(rule, 4)
        key(numerator)
        key(denominator)
        if number(low) > number(high):
            raise ValueError("Invalid ratio interval")
    for rule in compatible:
        left, right, allowed = sequence(rule, 3)
        key(left)
        key(right)
        if type(allowed) is not dict or any(
            type(name) is not str or type(options) is not list or
            any(type(option) is not str for option in options)
            for name, options in allowed.items()
        ):
            raise ValueError("Invalid compatibility rule")
    return WorldRules(**value)


def _selected_trace(path, index, trace_id):
    record_index = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if trace_id is not None:
                record = _json(line)
                if type(record) is not dict:
                    raise ValueError("JSONL records must be objects")
                if record.get("trace_id") == trace_id:
                    return trace_from_dict(record)
            elif record_index == index:
                return trace_from_dict(_json(line))
            record_index += 1
    raise ValueError("Selected record does not exist")


def _write_private_output(path, record, *, overwrite):
    # Build a complete file before publishing it. Exclusive hard-link creation
    # prevents a race from silently overwriting a file created during API calls.
    fd, temporary = tempfile.mkstemp(prefix=".synthetic-trace-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(record, target, ensure_ascii=False, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def parser():
    result = argparse.ArgumentParser(description=(
        "Run real model/API calls for one trace; the final attacker uses web search only with --web-search. "
        "Ground aliases must describe the selected input trace."
    ))
    result.add_argument("input", type=Path, help="Original source_traces.jsonl or canonical JSONL")
    result.add_argument("--ground", type=Path, required=True, help=(
        "Private ground aliases JSON; ground.example.json describes Microsoft/Activision only"
    ))
    for name in ("inference", "anonymizer", "generator", "attacker"):
        result.add_argument(f"--{name}-model", required=True, help="Explicit provider model ID")
    result.add_argument("--abstraction-rounds", type=int, default=2)
    result.add_argument("--outer-rounds", type=int, default=3)
    result.add_argument("--rules", type=Path, help="Optional JSON object with WorldRules fields")
    selector = result.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, help="Zero-based nonempty JSONL record index; default 0")
    selector.add_argument("--trace-id", help="Exact trace_id in the input, before import neutralizes IDs")
    result.add_argument("--out", type=Path, required=True, help="One successful synthetic trace as JSONL")
    result.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output")
    result.add_argument("--web-search", action="store_true", help="Let the final attacker use web search (off by default)")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    selected_index = 0 if args.index is None else args.index
    if selected_index < 0 or args.abstraction_rounds < 1 or args.outer_rounds < 0:
        print("Invalid index or round budget; no API calls made.", file=sys.stderr)
        return 2
    try:
        output = args.out.absolute()
        protected = [args.input, args.ground] + ([args.rules] if args.rules else [])
        if output.resolve() in {path.resolve() for path in protected}:
            print("Output must differ from every input file; nothing written.", file=sys.stderr)
            return 2
        if not output.parent.is_dir() or output.is_dir():
            print("Output parent must exist and output must be a file; nothing written.", file=sys.stderr)
            return 2
        if os.path.lexists(output) and not args.overwrite:
            print("Output already exists; use --overwrite to replace it. No API calls made.", file=sys.stderr)
            return 2
        trace = _selected_trace(args.input, selected_index, args.trace_id)
        ground = _ground(args.ground)
        rules = _rules(args.rules)
        # Construction is lazy: no SDK/client initialization occurs here.
        inference = PromptInferenceModel(OpenAIResponsesBackend(args.inference_model))
        anonymizer = PromptAnonymizerModel(OpenAIResponsesBackend(args.anonymizer_model))
        generator = PromptWorldGenerator(OpenAIResponsesBackend(args.generator_model))
        attacker = PromptFinalAttacker(OpenAIResponsesBackend(args.attacker_model),
                                       web_search=args.web_search)
        result = synthesize_trace(trace, ground, args.abstraction_rounds, args.outer_rounds,
            inference_model=inference, anonymizer_model=anonymizer,
            generator=generator, final_attacker=attacker, rules=rules,
            tool_validators=DEFAULT_TOOL_VALIDATORS, require_web_search=args.web_search)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        # Input or provider text may be present in exception strings. Keep all
        # such text out of stderr and do not persist failed inputs/feedback.
        print("Invalid or unreadable input/configuration; no output written.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; no output written.", file=sys.stderr)
        return 130
    print(json.dumps({"status": result.status, "reason": result.reason,
                      "attempts": [asdict(item) for item in result.attempts]}), file=sys.stderr)
    if not result.succeeded or result.trace is None:
        return 1
    try:
        _write_private_output(output, trace_to_dict(result.trace), overwrite=args.overwrite)
    except (OSError, UnicodeError, ValueError):
        print("Could not publish output; existing output was not intentionally removed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
