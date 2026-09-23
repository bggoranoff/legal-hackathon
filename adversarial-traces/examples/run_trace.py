"""Explicit, opt-in API runner for one or more selected JSONL traces, converted in parallel.

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
import time
from concurrent.futures import ThreadPoolExecutor

from adversarial_traces import (
    DEFAULT_TOOL_VALIDATORS, GroundTruth, WorldRules,
    synthesize_trace, trace_from_dict, trace_to_source_record,
)
from adversarial_traces.adapters import (
    PromptAnonymizerModel, PromptFinalAttacker, PromptInferenceModel, PromptMatchJudge,
    PromptWorldGenerator,
)
from adversarial_traces.matching import validate_ground
from adversarial_traces.bedrock_backend import BedrockConverseBackend, default_reasoning_effort
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


def _workflow(record):
    """The generic task type, kept so the output reads like the source traces."""
    workflow = record.get("workflow") if type(record) is dict else None
    return workflow if isinstance(workflow, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", workflow) else None


def _selected_traces(path, indexes, trace_ids, first):
    """Return [(source trace_id, Trace, workflow)] in the order requested."""
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = _json(line)
                if type(record) is not dict:
                    raise ValueError("JSONL records must be objects")
                records.append(record)
    if trace_ids:
        by_id = {record.get("trace_id"): record for record in records}
        if any(trace_id not in by_id for trace_id in trace_ids):
            raise ValueError("Selected record does not exist")
        chosen = [by_id[trace_id] for trace_id in trace_ids]
    elif first:
        chosen = records[:first]
    else:
        if any(index >= len(records) for index in indexes):
            raise ValueError("Selected record does not exist")
        chosen = [records[index] for index in indexes]
    if not chosen:
        raise ValueError("No records selected")
    return [(str(record.get("trace_id", i)), trace_from_dict(record), _workflow(record))
            for i, record in enumerate(chosen)]


def _write_private_output(path, records, *, overwrite):
    # Build a complete file before publishing it. Exclusive hard-link creation
    # prevents a race from silently overwriting a file created during API calls.
    fd, temporary = tempfile.mkstemp(prefix=".synthetic-trace-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            for record in records:
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


# Defaults: GPT 6 Luna (max reasoning) for stages 1-2 and the judge, GPT 6 Sol (high) attacks.
LUNA = "global.openai.gpt-6-luna"
SOL = "global.openai.gpt-6-sol"


def parser():
    result = argparse.ArgumentParser(description=(
        "Run real model/API calls for one trace. All models run on Amazon Bedrock, except the final "
        "attacker, which runs on OpenAI with web search when --web-search is given. "
        "Ground aliases must describe the selected input trace."
    ))
    result.add_argument("input", type=Path, help="Original source_traces.jsonl or canonical JSONL")
    result.add_argument("--ground", type=Path, help=(
        "Optional answer key JSON (ground.example.json describes Microsoft/Activision only). "
        "Without it, a judge model compares the attack with the original trace"
    ))
    for name, default in (("inference", LUNA), ("anonymizer", LUNA), ("generator", LUNA),
                          ("attacker", SOL), ("judge", LUNA)):
        result.add_argument(f"--{name}-model", help=(
            f"Bedrock model ID (default: {default}). With --web-search the attacker needs an "
            "explicit OpenAI model ID"))
    result.add_argument("--abstraction-rounds", type=int, default=2)
    result.add_argument("--outer-rounds", type=int, default=5,
        help="Most full attack rounds before giving up (default: 5)")
    result.add_argument("--rules", type=Path, help="Optional JSON object with WorldRules fields")
    selector = result.add_mutually_exclusive_group()
    selector.add_argument("--index", type=int, action="append", help=(
        "Zero-based nonempty JSONL record index; repeat for several (default: 0)"))
    selector.add_argument("--trace-id", action="append", help=(
        "Exact trace_id in the input; repeat for several traces"))
    selector.add_argument("--first", type=int, help="Convert the first N traces in the input")
    result.add_argument("--parallel-traces", type=int, help=(
        "How many traces to convert at once (default: all selected, up to 4)"))
    result.add_argument("--out", type=Path, required=True, help=(
        "JSONL file for the synthetic traces that pass, one per line, in the source_traces.jsonl "
        "format and in the order selected"))
    result.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output")
    result.add_argument("--web-search", action="store_true", help=(
        "Run the final attacker on OpenAI with web search (off by default; needs OPENAI_API_KEY)"))
    result.add_argument("--region", help="Bedrock region; defaults to AWS_REGION, then us-east-1")
    result.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
        help=("Reasoning level for the generator, attacker and judge "
              "(default: max for GPT 6 Luna, high for GPT 6 Sol, unset for others)"))
    result.add_argument("--stage1-reasoning", choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high", help=("Reasoning level for the stage 1 finder and rewriter, which make most of the "
                              "calls (default: high; ignored for models without a reasoning setting)"))
    result.add_argument("--parallel-steps", type=int, default=8,
        help="How many steps stage 1 rewrites at once (default: 8; 1 = one at a time)")
    result.add_argument("--bedrock-structured", choices=("tool", "text"), default="tool", help=(
        "How Bedrock returns JSON: forced tool call (default) or plain JSON text for models without tool choice"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.web_search and not args.attacker_model:
        print("--web-search needs an explicit OpenAI --attacker-model; no API calls made.", file=sys.stderr)
        return 2
    for name, default in (("inference_model", LUNA), ("anonymizer_model", LUNA),
                          ("generator_model", LUNA), ("attacker_model", SOL), ("judge_model", LUNA)):
        if not getattr(args, name):
            setattr(args, name, default)
    indexes = args.index or [0]
    if (any(index < 0 for index in indexes) or (args.first is not None and args.first < 1)
            or args.abstraction_rounds < 1 or args.outer_rounds < 0 or args.parallel_steps < 1
            or (args.parallel_traces is not None and args.parallel_traces < 1)):
        print("Invalid selection, round budget or parallelism; no API calls made.", file=sys.stderr)
        return 2
    try:
        output = args.out.absolute()
        protected = [args.input] + [path for path in (args.ground, args.rules) if path]
        if output.resolve() in {path.resolve() for path in protected}:
            print("Output must differ from every input file; nothing written.", file=sys.stderr)
            return 2
        if not output.parent.is_dir() or output.is_dir():
            print("Output parent must exist and output must be a file; nothing written.", file=sys.stderr)
            return 2
        if os.path.lexists(output) and not args.overwrite:
            print("Output already exists; use --overwrite to replace it. No API calls made.", file=sys.stderr)
            return 2
        selected = _selected_traces(args.input, indexes, args.trace_id, args.first)
        ground = _ground(args.ground) if args.ground else None
        if ground is not None and len(selected) > 1:
            print("An answer key (--ground) describes one trace; select one trace or drop --ground.",
                  file=sys.stderr)
            return 2
        rules = _rules(args.rules)
        # Construction is lazy: no SDK/client initialization occurs here.
        def bedrock(model, effort=None):
            # Without an explicit level the backend default applies (max for
            # GPT 6 Luna, high for GPT 6 Sol, none for models without one).
            effort = effort or args.reasoning_effort
            if effort and default_reasoning_effort(model) is None:
                effort = None  # e.g. Claude on Bedrock rejects a reasoning level
            kwargs = {"reasoning_effort": effort} if effort else {}
            return BedrockConverseBackend(model, region=args.region, structured=args.bedrock_structured, **kwargs)

        # Stage 1 makes most of the calls, so it runs at a lower level by default.
        inference = PromptInferenceModel(bedrock(args.inference_model, args.stage1_reasoning))
        anonymizer = PromptAnonymizerModel(bedrock(args.anonymizer_model, args.stage1_reasoning))
        generator = PromptWorldGenerator(bedrock(args.generator_model))
        # Bedrock has no built-in web search, so a web-searching attacker uses OpenAI.
        attacker_backend = (OpenAIResponsesBackend(args.attacker_model) if args.web_search
                            else bedrock(args.attacker_model))
        attacker = PromptFinalAttacker(attacker_backend, web_search=args.web_search)
        # No answer key: the judge (the only model shown the original) scores attacks.
        judge = None if ground is not None else PromptMatchJudge(bedrock(args.judge_model))
        # Model clients are shared: they are stateless and safe across threads.
        def convert(item):
            label, trace, workflow = item
            started = time.monotonic()
            try:
                result = synthesize_trace(trace, ground, args.abstraction_rounds, args.outer_rounds,
                    inference_model=inference, anonymizer_model=anonymizer,
                    generator=generator, final_attacker=attacker, rules=rules,
                    tool_validators=DEFAULT_TOOL_VALIDATORS, require_web_search=args.web_search,
                    judge=judge, parallel_steps=args.parallel_steps)
            except (ValueError, TypeError, RecursionError):
                # Exception text may include trace content; report the kind only.
                summary = {"trace": label, "status": "error", "reason": "invalid input or configuration"}
                print(json.dumps(summary), file=sys.stderr, flush=True)
                return None
            summary = {"trace": label, "status": result.status, "reason": result.reason,
                       "rounds": len(result.attempts), "seconds": round(time.monotonic() - started),
                       "attempts": [asdict(item) for item in result.attempts]}
            print(json.dumps(summary), file=sys.stderr, flush=True)
            if not result.succeeded or result.trace is None:
                return None
            return trace_to_source_record(result.trace, workflow=workflow)

        workers = min(args.parallel_traces or min(len(selected), 4), len(selected))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(convert, selected))
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        # Input or provider text may be present in exception strings. Keep all
        # such text out of stderr and do not persist failed inputs/feedback.
        print("Invalid or unreadable input/configuration; no output written.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; no output written.", file=sys.stderr)
        return 130
    passed = [record for record in records if record is not None]
    print(json.dumps({"passed": len(passed), "selected": len(selected)}), file=sys.stderr)
    if not passed:
        return 1
    try:
        _write_private_output(output, passed, overwrite=args.overwrite)
    except (OSError, UnicodeError, ValueError):
        print("Could not publish output; existing output was not intentionally removed.", file=sys.stderr)
        return 2
    return 0 if len(passed) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
