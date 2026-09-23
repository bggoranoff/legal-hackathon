# Orizon

**A clean room for AI agent traces in regulated industries.**

## The problem

Law firms (and other regulated businesses) are starting to use AI agents. Every agent run leaves a *trace*: the user's question, the agent's replies, the searches it ran and the documents it read. These traces are very useful for testing, debugging and improving agents, but they are full of confidential client information, so they can't be shared.

The usual fix is **redaction**: swap names, emails, dates and so on for placeholders like `[PERSON_1]`. Our claim is that this doesn't work. The rest of the trace (the deal, the terms, the timeline) is still there, and a capable AI with web access can often work out who the client is and fill the blanks back in.

## What we're building

Instead of redacting, Orizon makes **synthetic traces**: new traces that keep the same shape as the real ones (same steps, same tools, similar length and tone) but with every company, person, figure, date and quote replaced by made-up content. They are still useful for building and testing agents, but there is nothing real left to recover.

The demo shows this side by side:

1. **Original**: browse the real traces.
2. **Transform**: make a *redacted* copy or a *synthetic* copy.
3. **Reconstruct**: an "attacker" tries to rebuild the originals from that copy.
4. **Verify**: see how much of the real private info the attacker got back.

The expected result: redacted traces leak almost everything; synthetic traces leak almost nothing.

## The data

`backend/data/source_traces.jsonl` has 100 legal-agent traces: 10 real, public M&A deals × 10 legal tasks.

- **Deals:** Microsoft/Activision, Adobe/Figma, Cisco/Splunk, Amazon/iRobot, JetBlue/Spirit, Chiesi/Amryt, Renesas/Transphorm, Renesas/Sequans, Pelican/GSE, Altair/Datawatch.
- **Tasks:** payment structure, closing conditions, termination rights, approval risk, interim operations, competing offers, employee awards, document comparison, distinctive issues, deal team memo.

The conversations are simulated, but the agent's searches and reads ran against real public filings (e.g. SEC documents). Each trace has 10–25 steps.

## How it works

| File | What it does |
| --- | --- |
| `backend/data.py` | Loads the traces and flattens each one into a simple list of steps. |
| `backend/scrub.py` | **Redaction.** Finds private info using OpenAI's Privacy Filter model (plus a few exact patterns for emails, card numbers and secret keys) and swaps it for numbered placeholders. The same person always gets the same placeholder within a trace. |
| `backend/synth.py` | **Synthetic traces.** Sends real traces to Claude on Amazon Bedrock in one call and asks for fictional look-alikes. The result is cached. |
| `backend/score.py` | **The attack and the score.** An AI "investigator" reads a redacted or synthetic trace and guesses the real names, emails, dates, etc. Guesses are checked against what was actually removed. Based on the method from the paper *Beyond Memorization* (Staab et al.). Runs from the command line. |
| `backend/traces.py` | Ties the above together for the API. |
| `backend/app.py` | The web API (FastAPI). |
| `frontend/` | The dashboard (React + Vite). |

### API

| Endpoint | Returns |
| --- | --- |
| `GET /api/traces/original` | The real traces. |
| `POST /api/redact` | Redacted copies. |
| `POST /api/synthetic` | Synthetic copies. |
| `POST /api/reconstruct` `{mode}` | The attacker's rebuilt traces. |
| `POST /api/verify` `{mode}` | Share of real private info recovered. |

`mode` is `"redact"` or `"synthetic"`.

## What's real and what's still mocked

- ✅ **Redaction** is real (Privacy Filter model).
- ✅ **Synthetic generation** is real (Bedrock), but only for the first 8 traces (`SYNTH_SAMPLE`) in a single call.
- ✅ **The scorer** (`score.py`) is real, but it only runs from the command line. It isn't connected to the dashboard yet.
- ⚠️ **Reconstruct** in the dashboard is faked: for redacted traces it just returns the originals, and for synthetic traces it returns the synthetic ones.
- ⚠️ **Verify** in the dashboard is hard-coded to 95% (redacted) and 1% (synthetic).

## Running it

### With Docker

Put your keys in a `.keys` file in the repo root (it's git-ignored):

```bash
AWS_BEARER_TOKEN_BEDROCK=...
AWS_REGION=us-east-1
```

Then:

```bash
./deploy.sh
```

- Dashboard: http://localhost:3000
- API: http://localhost:8000 (docs at `/docs`)

### Without Docker

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Running the scorer

```bash
cd backend
python score.py --mode both --limit 5   # modes: redact, synthetic, both
```

Prints the share of private info the attacker recovered, overall and by type (names, emails, dates, ...).

### Settings

| Variable | Default | Used for |
| --- | --- | --- |
| `AWS_BEARER_TOKEN_BEDROCK` | none | Bedrock access (synthetic traces and scorer) |
| `AWS_REGION` | `us-east-1` (scorer: `eu-central-1`) | Bedrock region |
| `BEDROCK_MODEL_ID` | Claude 3.5 Sonnet | Model that writes synthetic traces |
| `SYNTH_SAMPLE` | `8` | How many traces to make synthetic copies of |

**Note:** the first redaction downloads the Privacy Filter model (about 2.8 GB) and is slow without a GPU.
