# Orizon — a clean room for AI-agent logs in regulated industries

Legal, healthcare and finance teams increasingly run LLM agents over privileged
material, and every run leaves a trace: the prompts, the tool calls, the retrieved
documents, the drafted memos. Those traces are exactly what you want for evals,
debugging and model improvement — and exactly what you cannot share, because they are
full of client identities, matter details, and regulated PII.

The usual answer is to **strip the PII**. This demo shows why that is not enough, and
what to do instead.

## The thesis

1. **PII stripping does not protect the client.** Redaction removes the surface tokens
   (names, emails, dates) but leaves the *structure* intact — the deal terms, the
   sector, the figures, the sequence of events. A capable LLM reads the redacted trace
   and **reconstructs** who it is about. In our M&A dataset the adversary recovers the
   real parties from redacted traces at a high rate.

2. **Fully synthetic traces do protect the client.** Instead of deleting from the real
   trace, we *evolve* a new one: abstract every fact to a placeholder, invent a coherent
   fictional world, fill the trace with it, and then have a **separate attacker model try
   to trace it back to the original matter**. We keep only the traces the attacker fails
   to re-identify. Because nothing real underlies the result, the same adversary recovers
   ≈ 0. The trace keeps its shape — same tools, same steps, same JSON — so it is still
   useful for evals.

3. **Future work: a provable guarantee.** Today's synthetic guarantee is *empirical* — a
   strong attacker did not re-identify the trace, which is not a proof of anonymity. The
   plan is to tighten it with **differential privacy** on the generation step, turning
   "an attacker failed" into a formal bound.

The whole demo is a four-step loop that lets you watch (1) fail and (2) succeed on the
same real traces, side by side.

## The demo loop

```
Original traces ──► Transform ──► Reconstruct ──► Verify
                    │  Redact  │   (adversary      (adversarial
                    │  Synthetic│    LLM guesses     accuracy:
                    └───────────┘    the real deal)   % recovered)
```

The dashboard shows four aligned columns for the same matters:

| Step | What happens | Backend |
| --- | --- | --- |
| **Original** | The real corporate M&A agent traces, PII highlighted. | `data.py`, `traces.py` |
| **Transform → Redact** | Every PII span replaced by a consistent placeholder (`[PERSON_1]`, `[ORG_1]`, …). | `scrub.py`, `traces.redact_traces` |
| **Transform → Synthetic** | A fresh trace evolved through the adversarial pipeline; no real content survives. | `synth.py` → `adversarial-traces/` |
| **Reconstruct** | An adversary LLM sees only the transformed trace and guesses the real companies + deal year, filling the blanks with its guess. | `traces._infer`, `score.py` |
| **Verify** | Reports **adversarial accuracy** — the share of the real deal identity the adversary recovered. High on redacted, ≈ 0 on synthetic. | `score.py`, `traces.verify_score` |

Redaction leaves the `[ORG_n]` blanks that the attacker fills back in; synthetic traces
carry no real deal to recover, so there is nothing to fill.

## How each half works

### Redaction (the baseline that fails)

`scrub.py` is ported from `TokenTrim/orizon-scrub`. Detection runs OpenAI's
[Privacy Filter](https://github.com/openai/privacy-filter) (`opf`) model at a
**high-recall** operating point, augmented with high-precision regex for the structured
types the model is unreliable on (emails, Luhn-checked cards, secret keys). Each entity
maps to a consistent numbered placeholder within a trace, and party company names are
additionally collapsed to `[ORG_1]`, `[ORG_2]`. It is a genuine, aggressive redactor —
and the adversary still wins, because the deal structure is untouched.

### Synthetic generation (the approach that works)

`synth.py` drives the **`adversarial-traces/`** package (its own README has the full
API). For each real trace:

1. **Abstract** — `adversarial_anonymize` runs an infer-then-rewrite loop: a local model
   infers identifying attributes from the current text, an anonymizer rewrites to remove
   them, repeated for K rounds. Every fact becomes a stable placeholder (`{{BUYER}}`,
   `{{PRICE}}`, `{{SIGNING_DATE}}`).
2. **Generate one world** — a generator invents a single coherent fictional binding for
   all placeholders, subject to `WorldRules` (date ordering, numeric ranges, jurisdiction
   ↔ governing-law compatibility).
3. **Fill** — the world is applied deterministically across every segment, so the same
   fact reads consistently throughout. Structure (roles, tool-call pairing, JSON shape)
   is owned by the program, never by a model.
4. **Attack** — a **different** attacker model (a required `model_id` mismatch enforces
   this) tries to name the original matter from the finished trace. A local, deterministic
   matcher checks the answer against a held-out ground-truth key that no model ever sees.
5. **Keep or retry** — a trace is emitted only if the attacker fails. Re-identified traces
   feed a hint into the next round; a real name left in the trace fails immediately.

Successful traces are written to the same JSONL format as the source, so the dashboard
loads them directly. `synth.py` builds incrementally and caches each success as it lands.

### The scorer

`score.py` is the re-identification scorer, ported from the ETH-SRI work (see
[References](#references)). An adversary LLM reads the transformed trace and guesses the
removed PII; each guess is matched to ground truth by Jaro-Winkler similarity (> 0.75)
with a model judge as fallback (`yes` / `no` / `less precise` → 1 / 0 / 0.5). The reported
number is the paper's **adversarial accuracy**: the fraction of real PII recovered. A
*high* score on redacted traces is the point — the stripping did not protect anyone.

- The dashboard's **Verify** measures deal-identity recovery (real companies + announcement
  year), the signal that matters for M&A.
- `python score.py --mode {redact,synthetic,both}` runs the fuller per-PII-category scorer
  over `opf`-derived ground truth, standalone.

Both use `gpt-5.6-luna` on Amazon Bedrock as the adversary/judge.

## Repository layout

```
backend/                 FastAPI demo API (app.py) + transforms
  data.py                Load & flatten the M&A source traces
  traces.py              Original / redact / synthetic / reconstruct / verify
  scrub.py               opf + regex PII detection and placeholder replacement
  synth.py               Drives the adversarial-traces pipeline (Bedrock)
  score.py               Adversarial re-identification scorer (the paper's metric)
  data/source_traces.jsonl   100 traces across 10 real M&A deals
adversarial-traces/      Standalone package: adversarial_anonymize + synthesize_trace
frontend/                React + Vite dashboard (the four-column view)
docker-compose.yml       backend :8000, frontend :3000
deploy.sh                Build & run both with Docker
```

The dataset is 100 agent traces over 10 public M&A matters (Microsoft/Activision,
Adobe/Figma, Cisco/Splunk, …), each a full session of messages and tool calls
(`search_documents`, `read_section`, `compare_documents`, `draft_memo`).

## Running it

```sh
./deploy.sh            # loads .keys, then docker compose up --build -d
# Dashboard : http://localhost:3000
# API       : http://localhost:8000  (docs at /docs)
```

Bedrock access is required for the synthetic pipeline and the scorer. Put
`AWS_BEARER_TOKEN_BEDROCK` in `.keys` (gitignored) and set `AWS_REGION=eu-central-1`.
`boto3` needs the `awscrt` package for bearer auth. Relevant knobs: `SYNTH_SAMPLE` /
`DEMO_LIMIT` (how many traces to process) and `CACHE_SUFFIX` (build caches off to the
side). Redacted and synthetic results are cached under `backend/data/`, so a first run is
slow (the `opf` model is ~2.8 GB and generation is many Bedrock calls) and later runs are
instant.

Without Docker:

```sh
cd backend && pip install -r requirements.txt && pip install -e ../adversarial-traces
uvicorn app:app --reload            # backend
cd ../frontend && npm install && npm run dev   # dashboard
```

## References

The re-identification adversary and the anonymization loop both come from Martin Vechev's
group at ETH SRI, and the code lives in
[`eth-sri/llm-anonymization`](https://github.com/eth-sri/llm-anonymization).

- Staab, Vero, Balunović, Vechev — **Large Language Models are Advanced Anonymizers**,
  ICLR 2025. [arXiv:2402.13846](https://arxiv.org/abs/2402.13846). The abstract →
  invent-a-world → attack loop we use for synthetic generation follows this method; the
  coherent whole-trace generation is our extension of it.
- Staab, Vero, Balunović, Vechev — **Beyond Memorization: Violating Privacy via Inference
  with Large Language Models**, ICLR 2024. The re-identification adversary and the
  "adversarial accuracy" metric in `score.py` are from this companion paper.

Our synthetic guarantee is empirical (a configured attacker failed to re-identify), **not**
a differential-privacy proof — see [thesis point 3](#the-thesis).
