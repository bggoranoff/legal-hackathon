"""Generate synthetic traces through the adversarial_traces pipeline, one at a time, cached to disk.

`generate` is the fast serve path (reads the cache). `generate_all` is the slow builder: it walks
the deals round-robin, keeps only traces the adversary fails to re-identify, and writes each success
to the cache the moment it lands — so a partial run still leaves usable traces for the demo.
"""

import json
import os
import sys
import time

import data

from adversarial_traces import (
    DEFAULT_TOOL_VALIDATORS, GroundTruth, WorldRules,
    synthesize_trace, trace_from_dict, trace_to_source_record,
)
from adversarial_traces.adapters import (
    PromptAnonymizerModel, PromptFinalAttacker, PromptInferenceModel, PromptWorldGenerator,
)
from adversarial_traces.bedrock_backend import BedrockConverseBackend

_SAMPLE = int(os.environ.get("SYNTH_SAMPLE", os.environ.get("DEMO_LIMIT", "10")))  # target successes
_CACHE = os.path.join(os.path.dirname(__file__), "data", f"synthetic_traces{os.environ.get('CACHE_SUFFIX', '')}.json")
_REGION = os.environ.get("AWS_REGION", "eu-central-1")
# gpt-5.6-luna is the model verified on this AWS key; the pipeline needs the final attacker on a
# different model_id from local inference, so the attacker uses the sol sibling.
_MODEL = "global.openai.gpt-5.6-luna"
_ATTACKER = "global.openai.gpt-5.6-sol"


def _log(msg):
    print(f"[synth] {msg}", file=sys.stderr, flush=True)


def _bedrock(model):
    return BedrockConverseBackend(model, region=_REGION, structured="tool")


def _ground(record):
    """Build the private scoring ground truth for a raw record from its case name."""
    case = record["case_id"].rsplit("_", 1)[0]
    name = data._CASES.get(case, case.replace("_", " ").title())
    parties = [p.strip() for p in name.split(" / ")]
    return GroundTruth(identities=(f"{name} deal",), party_aliases=tuple((p, p) for p in parties))


def _synthesize(record):
    """Return one synthetic source record for a raw record, or None if the attacker re-identified it."""
    result = synthesize_trace(
        trace_from_dict(record), _ground(record), 2, 3,
        inference_model=PromptInferenceModel(_bedrock(_MODEL)),
        anonymizer_model=PromptAnonymizerModel(_bedrock(_MODEL)),
        generator=PromptWorldGenerator(_bedrock(_MODEL)),
        final_attacker=PromptFinalAttacker(_bedrock(_ATTACKER), web_search=False),
        rules=WorldRules(), tool_validators=DEFAULT_TOOL_VALIDATORS, require_web_search=False)
    if not result.succeeded:
        return None
    return trace_to_source_record(result.trace, workflow=record.get("workflow"))


def _ordered_records():
    """Raw records interleaved round-robin across deals, so obscure (passable) deals come up early."""
    with open(data._PATH) as f:
        records = [json.loads(line) for line in f]
    groups = {}
    for record in records:
        groups.setdefault(record["case_id"], []).append(record)
    order = []
    for i in range(max(len(v) for v in groups.values())):
        for case in groups:
            if i < len(groups[case]):
                order.append(groups[case][i])
    return order


def _write(out):
    tmp = _CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, _CACHE)  # atomic, so a concurrent reader never sees a half file


def generate(originals):
    """Serve path: return the cached synthetic set (whatever the builder has written so far)."""
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            return json.load(f)
    return []


def generate_all():
    """Build the synthetic set incrementally, writing each success to the cache as it lands."""
    records = _ordered_records()
    _log(f"target {_SAMPLE} synthetic traces from {len(records)} deals (attacker skips re-identifiable ones)")
    out = []
    for n, record in enumerate(records, 1):
        rid = record["trace_id"]
        _log(f"{n}/{len(records)} {rid}: synthesizing…")
        start = time.time()
        try:
            synthetic = _synthesize(record)
        except Exception as exc:
            _log(f"{n}/{len(records)} {rid}: ERROR {type(exc).__name__}: {exc}")
            continue
        took = time.time() - start
        if synthetic is None:
            _log(f"{n}/{len(records)} {rid}: re-identified, skipped ({took:.0f}s)")
            continue
        flat = data.flatten_record(synthetic)
        flat["id"] = f"syn-{len(out) + 1:03d}"
        flat["source_id"] = rid
        out.append(flat)
        _write(out)
        _log(f"{n}/{len(records)} {rid}: OK -> {flat['id']} ({took:.0f}s); {len(out)}/{_SAMPLE} written")
        if len(out) >= _SAMPLE:
            break
    _log(f"done: {len(out)} synthetic traces written to {_CACHE}")
    return out


if __name__ == "__main__":
    generate_all()
