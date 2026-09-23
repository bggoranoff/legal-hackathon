"""Load the corporate M&A legal-agent traces and flatten them to the demo's step shape."""

import json
import os

_PATH = os.path.join(os.path.dirname(__file__), "data", "source_traces.jsonl")
_MAX = 600  # cap any single step detail so document passages stay readable in the pane

_CASES = {
    "microsoft_activision": "Microsoft / Activision Blizzard",
    "adobe_figma": "Adobe / Figma",
    "cisco_splunk": "Cisco / Splunk",
    "amazon_irobot": "Amazon / iRobot",
    "jetblue_spirit": "JetBlue / Spirit",
    "chiesi_amryt": "Chiesi / Amryt",
    "renesas_transphorm": "Renesas / Transphorm",
    "renesas_sequans": "Renesas / Sequans",
    "pelican_gse": "Pelican / GSE",
    "altair_datawatch": "Altair / Datawatch",
}


def _clip(text):
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX else text[:_MAX].rstrip() + " …"


def _detail(event):
    kind = event["event_type"]
    if kind == "message":
        return _clip(event["content"])
    if kind == "tool_call":
        args = event.get("arguments", {})
        return _clip(args.get("query") or args.get("chunk_id") or json.dumps(args))
    result = event.get("result", {})
    if event["tool_name"] == "search_documents":
        hits = ", ".join(r["chunk_id"] for r in result.get("results", [])[:3])
        return _clip(f"{result.get('matched_chunks', 0)} matches — top: {hits}")
    if "text" in result:
        return _clip(result["text"])
    return _clip(json.dumps(result))


def _action(event):
    if event["event_type"] == "message":
        return event["role"]
    verb = "call" if event["event_type"] == "tool_call" else "result"
    return f"{event['tool_name']} {verb}"


def _matter(record):
    case = record["case_id"].rsplit("_", 1)[0]
    name = _CASES.get(case, case.replace("_", " ").title())
    return f"{name} — {record['workflow'].replace('_', ' ').title()}"


def load_traces():
    """Return every session as {id, matter, steps:[{step, action, detail}]}."""
    out = []
    with open(_PATH) as f:
        for line in f:
            record = json.loads(line)
            steps = [
                {"step": e["sequence"], "action": _action(e), "detail": _detail(e)}
                for e in record["events"]
            ]
            out.append({"id": record["trace_id"], "matter": _matter(record), "steps": steps})
    return out
