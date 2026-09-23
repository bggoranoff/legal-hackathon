"""Generate synthetic traces once, via a single Amazon Bedrock call, from the real traces."""

import json
import os

import boto3
from botocore.config import Config

_MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
_SAMPLE = int(os.environ.get("SYNTH_SAMPLE", "8"))  # traces sent/generated in the one call

_PROMPT = (
    "You generate SYNTHETIC corporate legal-agent traces for a privacy demo. "
    "Given real M&A legal-agent traces, output the SAME NUMBER of synthetic traces that keep the "
    "same structure — same number of steps, same `action` names, similar length and tone — but "
    "replace every real company, person, deal, figure, date and quoted document passage with "
    "plausible but entirely fictional content. No real entity may be recoverable from the output. "
    "Return ONLY a JSON array, no prose, no code fences. Each element: "
    '{"id":"syn-001","matter":str,"steps":[{"step":int,"action":str,"detail":str}]}. '
    "Number ids syn-001, syn-002, … and keep each trace's step count equal to its input."
)

_cache = None


def _trim(traces):
    """Compact the sampled originals so the prompt stays within one call's budget."""
    out = []
    for t in traces[:_SAMPLE]:
        steps = [{"step": s["step"], "action": s["action"], "detail": s["detail"][:300]} for s in t["steps"]]
        out.append({"id": t["id"], "matter": t["matter"], "steps": steps})
    return out


def _parse(text):
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])


def generate(originals):
    """Return the cached synthetic set, calling Bedrock once on first use."""
    global _cache
    if _cache is not None:
        return _cache
    client = boto3.client("bedrock-runtime", region_name=_REGION, config=Config(read_timeout=120))
    resp = client.converse(
        modelId=_MODEL,
        messages=[{"role": "user", "content": [
            {"text": _PROMPT + "\n\nREAL TRACES:\n" + json.dumps(_trim(originals))}
        ]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 0.7},
    )
    _cache = _parse(resp["output"]["message"]["content"][0]["text"])
    return _cache
