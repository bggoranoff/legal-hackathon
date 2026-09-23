"""Replace the adobe_figma synthetic slots with the full-Vechev traces from the 9-of-10 file."""

import collections
import json
import re
import sys

import data

CACHE = "data/synthetic_traces.json"
NINE = "/Users/bggoranoff/Downloads/adobe_figma_synthetic_9_of_10.jsonl"

_ORG_RE = re.compile(
    r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3},?\s+"
    r"(?:Inc|Ltd|LLC|L\.P|Corp|Corporation|Holdings|Systems|Robotics|Instruments|"
    r"Analytics|Technologies|Group|Industries|Solutions)\b\.?"
)


def _parties(flat):
    """The two most frequent company-shaped names across the trace, for the matter label."""
    text = " ".join(s["detail"] for s in flat["steps"])
    counts = collections.Counter()
    for m in _ORG_RE.finditer(text):
        counts[re.sub(r"[.,]+$", "", m.group(0)).strip()] += 1
    top = [name for name, _ in counts.most_common(2)]
    return top


cache = json.load(open(CACHE))
nine = [json.loads(line) for line in open(NINE) if line.strip()]

# workflow title -> the cache slot (an adobe_figma synthetic) that ends with it
adobe = {t["matter"].split(" — ")[-1]: i for i, t in enumerate(cache) if (t.get("source_id") or "").startswith("adobe_figma")}

replaced = []
for record in nine:
    title = record["workflow"].replace("_", " ").title()
    if title not in adobe:
        print(f"! no adobe slot for workflow {record['workflow']}", file=sys.stderr)
        continue
    idx = adobe[title]
    slot = cache[idx]
    flat = data.flatten_record(record)
    parties = _parties(flat)
    matter = f"{' / '.join(parties)} — {title}" if len(parties) == 2 else slot["matter"]
    cache[idx] = {"id": slot["id"], "source_id": slot["source_id"], "matter": matter, "steps": flat["steps"]}
    replaced.append(f"{slot['id']} <- {slot['source_id']} | {matter}")

json.dump(cache, open(CACHE, "w"))
print(f"replaced {len(replaced)} of {len(cache)} traces:")
for r in replaced:
    print("  ", r)
