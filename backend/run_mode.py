"""Compute + cache one transform's inference pipeline and print its score. Usage: run_mode.py {redact|synthetic}"""

import json
import sys

import traces

mode = sys.argv[1]
score = traces.verify_score(mode)
recon = traces.reconstruct_traces(mode)
filled = [st["detail"] for t in recon for st in t["steps"] if "\u0001" in st["detail"]]
print(json.dumps({"mode": mode, "score": score, "traces": len(recon), "sample_filled": filled[:3]}, ensure_ascii=False))
