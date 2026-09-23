# adversarial-traces

A Python package implementing the two supplied algorithms:

- **`adversarial_anonymize`**: infer attributes from the current text, then rewrite using those inferences, for at most K rounds.
- **`synthesize_trace`**: abstract each segment, generate one invented world, fill it consistently across the trace, and ask a separate attacker model to identify the original matter. Web search for that attacker is off by default.

The core has no runtime dependencies. Model roles are pluggable Python protocols; an optional OpenAI Responses backend and structured prompt adapters are included. This is a new implementation of the supplied pseudocode, not a fork of the paper's code or a reproduction of its experiments.

**A successful result means the configured attacker did not match the supplied answer key. This package does not implement differential privacy or prove that a trace is anonymous.**

## Install

From this directory:

```bash
python -m pip install .
```

For the included provider adapters (Bedrock for everything, OpenAI only for a web-searching attacker):

```bash
python -m pip install '.[bedrock]'          # default
python -m pip install '.[bedrock,openai]'   # if you turn on web search
```

Python 3.10 or later is required. `src/adversarial_traces/` can also be added to an existing repository that uses a `src` package layout. Nothing is published to a package registry.

## Run the offline example

```bash
python examples/offline_demo.py
```

The example uses explicitly scripted model doubles. It exercises a first-round identification, feedback, a second invented world, and a passing result. It makes no model or web requests; its simulated attack outcome is a control-flow demonstration, not privacy evidence.

## Anonymize one text

Supply objects implementing the model protocols in `models.py`:

```python
from adversarial_traces import adversarial_anonymize

rewritten = adversarial_anonymize(
    text,
    {"matter_identity", "parties", "amounts", "dates"},
    3,
    inference_model=inference_model,
    anonymizer_model=anonymizer_model,
)
```

`infer(text, attributes)` receives only the current text and the requested attribute names. It returns `Inference` objects containing a guess, a short evidence explanation, certainty, and optional source spans. Only requested, non-null guesses are passed to `anonymize(text, inferences, hints=...)`.

No history is supplied. Empty inferences stop the ordinary inner loop. K=0 returns the input without calling either model. There is no extra attack after the final rewrite; the synthesis loop performs its own final whole-trace attack.

The prompt adapter instructs the anonymizer to remove, obfuscate or generalize, and never invent replacement facts. An optional `rewrite_validator(before, after)` callback can reject rewrites. Structure checks and prompt instructions alone cannot prove that a free-text rewrite introduced no new factual claims.

## Synthesize a trace

```python
from adversarial_traces import (
    GroundTruth, synthesize_trace, trace_from_dict, trace_to_source_record,
)

trace = trace_from_dict(source_record)
ground = GroundTruth(
    identities=("Microsoft acquisition of Activision Blizzard",),
    party_aliases=(
        ("Microsoft", "Microsoft Corporation"),
        ("Activision Blizzard", "Activision Blizzard, Inc."),
    ),
)

result = synthesize_trace(
    trace, ground,
    max_abstraction_rounds=3,
    max_outer_rounds=2,
    inference_model=inference_model,
    anonymizer_model=anonymizer_model,
    generator=generator,
    final_attacker=final_attacker,
)

if result.succeeded:
    synthetic_record = trace_to_source_record(result.trace, workflow=source_record["workflow"])
else:
    assert result.trace is None
    print(result.reason)  # sanitized category, no rejected trace or answer key
```

`trace_to_source_record` writes the result in the same format as `backend/data/source_traces.jsonl`: the same top-level fields and the same fields on every event. The repo's dashboard loader (`backend/data.py`) reads it directly. It gets a fresh `synthetic_<id>` case and trace ID, and keeps only the generic `workflow` (e.g. `structure_payment`). Timestamps, run times, `as_of_date` and `final_outcome` are null, and `artifacts` is empty, rather than copied from the real trace. `trace_to_dict` still gives the package's own internal format.

`AdversarialAnonymize` and `SynthesizeTrace` are also exported as aliases matching the pseudocode's names. Python argument names and structured return types are documented above.

### What each stage receives

| Component | Input |
| --- | --- |
| Local inference model | Current segment text and target attributes |
| Anonymizer | Current text, current inferences, applicable whole-trace hints |
| Generator | Abstracted trace, required placeholder names, trusted coherence rules |
| Final attacker | Filled synthetic trace and an instruction to find its original matter |
| Local matcher | Final guesses and `GroundTruth` |

`GroundTruth` is never supplied to any model. Model-facing traces contain fresh neutral envelope IDs and omit imported case labels, timestamps and private evaluation metadata. Document URLs, IDs and names **inside payloads** remain material for the anonymizer to process.

The library separates function inputs; it does not create isolated machines or a network boundary. Run inference and anonymization with clients approved for the source environment. A custom `JSONBackend` can call a local model; the included hosted-provider example is convenient for the public demo documents. An abstracted profile can still contain identifying clues and should not be treated as a proven-safe export.

### Whole-trace feedback

Instead of literally concatenating `segment + hints`, hints are delivered separately to the anonymizer. This avoids copying attacker explanations into generated text. A first rewrite is performed when whole-trace hints exist even if the local attacker finds nothing; otherwise the local empty-inference stop would ignore cross-step leaks. Hints are not given to local inference or to the generator as metadata.

Every outer round starts again from the original neutralized trace. After a matched attack, its spans and brief evidence explanation become guidance for the next pass. The original remains unchanged.

### One invented world

The anonymizer leaves stable placeholders such as `{{BUYER}}`, `{{TARGET}}`, `{{PRICE}}` and `{{SIGNING_DATE}}`. It must use the same semantic key for the same fact in every segment. The generator produces one `SyntheticWorld.bindings` map for the entire candidate, and `fill_trace` applies it deterministically throughout requests, tool arguments, results and answers.

An exact whole-value placeholder can restore a JSON number or boolean; an inline placeholder is rendered as text. Missing or extra bindings, nested placeholders, invalid scalar values and key collisions are rejected. Final payload types must agree with the original payload types.

For domain checks, pass `WorldRules`:

```python
from adversarial_traces import WorldRules

rules = WorldRules(
    date_orders=(("SIGNING_DATE", "CLOSING_DATE"),),
    numeric_ranges={
        "PRICE": (1_000_000, 1_000_000_000),
        "ANNUAL_REVENUE": (1_000_000, 1_000_000_000),
    },
    ratio_ranges=(("PRICE", "ANNUAL_REVENUE", 0.1, 20.0),),
    compatible_values=(
        ("JURISDICTION", "GOVERNING_LAW", {
            "Example jurisdiction A": ("Example law A",),
            "Example jurisdiction B": ("Example law B",),
        }),
    ),
)
```

These are illustrative constraints, not valuation guidance or a legal compatibility database. Rules must refer to placeholders present in the abstracted trace. Default checks validate canonical signing/closing/termination date ordering when those keys appear. Other price/size and jurisdiction/law relationships need caller-supplied rules. A `candidate_validator(trace)` callback can add checks across the complete candidate and raise `TraceError` to reject it before the attack.

### Valid trace structure

Canonical traces consist of ordered `Segment` objects with these payloads:

| Kind | Role | Payload |
| --- | --- | --- |
| `message` | `system`, `user` or `assistant` | `{"text": "..."}` |
| `tool_call` | `assistant` | `{"arguments": {...}}` |
| `tool_result` | `tool` | `{"result": ...}` |

Each tool result must match a preceding call's ID and tool name. IDs, order, roles, kinds, object fields and array lengths are preserved through rewriting. Only payload values change. Synthetic tool records are not executed against live systems.

The synthesis function includes argument validators for the earlier dataset's `search_documents`, `read_section`, `compare_documents` and `draft_memo` tools. Supply `tool_validators={"your_tool": validator}` for other tool schemas or to override a default. Validators receive argument dictionaries and raise on rejection. Preserving JSON structure alone does not establish that an arbitrary tool request is executable or that its result is semantically correct.

Sensitive dynamic dictionary keys or sensitive tool names require explicit normalization before synthesis: this implementation preserves those structural fields rather than silently renaming a tool API. Placeholder filling of dictionary keys is supported as a standalone helper, but changing object fields during the synthesis rewrite is rejected.

### Matching and failures

The default local matcher fails closed. The attack counts as a re-identification if **any** real party alias or matter identity is mentioned **anywhere** in the attacker's answer: any guess's identity or parties, its reasoning, or its cited spans. One party is enough, so "Microsoft's gaming buyout" matches a key that lists `Microsoft`.

Matching is fuzzy. Text is normalized for Unicode, capitalization, punctuation and spacing. An alias then matches if it appears as whole words, with different spacing ("Jet Blue" for "JetBlue"), or, for aliases of 5+ letters, as a close misspelling over the same number of words (at least 85% similar, `FUZZY_THRESHOLD` in `matching.py`), such as "Activison". Shorter aliases like tickers ("ADBE", "GSE") must appear exactly, because fuzzy matching them hits ordinary legal text such as "(a) be". No model judge is used. Certainty does not discard a guess.

The same check runs on the finished synthetic trace **before** the attack: if a real name is still in it, the round fails straight away and the next round is told to remove real names (without being told which, so the answer key never reaches a model).

Use an answer key for the selected trace, including likely buyer/target aliases and tickers (e.g. `MSFT`); fuzzy matching won't link a ticker to a name. Avoid aliases that are ordinary words (say, `Spirit` on its own), because any use of the word will then count as a match. Too few aliases can still produce a false pass. A custom `matcher(report, ground)` must remain local and deterministic to preserve the stated ground-truth boundary.

| Result | Meaning |
| --- | --- |
| `passed_attack` | A completed attack did not match the supplied ground truth; `trace` contains the candidate |
| `failed`, `reidentified` | The attacker kept naming the original, or a real name stayed in the trace, within the allowed rounds |
| `failed`, `invalid_candidate` | Rewriting, filling, or coherence failed validation within the allowed rounds |
| `failed`, `invalid_attack` | The final attack did not complete, or web search was required but not confirmed |
| `failed`, `model_error` | A provider or structured-response failure prevented evaluation |
| `failed`, `budget_exhausted` | No outer rounds were allowed |

Invalid input/configuration raises `TraceError`. Unexpected exceptions in custom code propagate; they are not interpreted as privacy success. Provider failures, refusals and truncated JSON are never converted to empty guesses. Rejected traces, attacker reasoning, private hints and answer keys are not returned in `SynthesisResult`. Attempt summaries contain only round numbers and sanitized statuses.

Synthesis requires a positive abstraction budget and a final attacker whose declared model ID differs from local inference. Its strength is a caller choice.

### Web search (off by default)

By default the final attacker works only from the trace and its own knowledge. To let it search the web, turn it on in both places:

```python
final_attacker = PromptFinalAttacker(backend, web_search=True)
result = synthesize_trace(..., final_attacker=final_attacker, require_web_search=True)
```

With `require_web_search=True`, an attack without a confirmed completed search fails as `invalid_attack`. A custom final attacker must honestly report completed web-search execution; the included OpenAI backend derives that flag from actual completed search output, not from model-written JSON. Bedrock has no built-in web search, so use `OpenAIResponsesBackend` for the attacker when this is on. The example runner takes `--web-search` for the same thing.

## Use the optional API adapters

Everything runs on Amazon Bedrock by default. OpenAI is used only for the final attacker when web search is on.

```python
from adversarial_traces.adapters import (
    PromptInferenceModel, PromptAnonymizerModel,
    PromptWorldGenerator, PromptFinalAttacker,
)
from adversarial_traces.bedrock_backend import BedrockConverseBackend
from adversarial_traces.openai_backend import OpenAIResponsesBackend

inference_model = PromptInferenceModel(BedrockConverseBackend(inference_model_id))
anonymizer_model = PromptAnonymizerModel(BedrockConverseBackend(anonymizer_model_id))
generator = PromptWorldGenerator(BedrockConverseBackend(generator_model_id))
final_attacker = PromptFinalAttacker(BedrockConverseBackend(attacker_model_id))

# With web search on instead:
# final_attacker = PromptFinalAttacker(OpenAIResponsesBackend(openai_model), web_search=True)
```

**Bedrock** uses the Converse API. It signs in the usual boto3 way, including `AWS_BEARER_TOKEN_BEDROCK`, and reads the region from `region=` or `AWS_REGION` (default `us-east-1`). JSON answers come back through a forced tool call. For a model that doesn't support that, pass `structured="text"` to ask for plain JSON instead. For reasoning models that take a reasoning level, such as GPT 6 Luna (`global.openai.gpt-6-luna`), pass `reasoning_effort=` one of `none`, `low`, `medium`, `high`, `xhigh`, `max` (the runner's `--reasoning-effort`). By default GPT 6 Luna runs at `max` and GPT 6 Sol at `xhigh` (both with a 32,000-token output limit and a 10-minute timeout to leave room for reasoning); other models get no reasoning setting, since Claude on Bedrock rejects it. Pass `reasoning_effort=None` for the model's own default. Each call is fresh, with no history. Asking the Bedrock backend for web search raises an error rather than silently skipping it. Provider errors report only the AWS error code, never the message, since messages can echo the trace.

**OpenAI** uses the Responses API with `store=False`, no conversation or prior-response ID, and no SDK retries. `store=False` does not itself promise provider zero retention.

Choose model IDs your account can use; all need structured outputs. There are no hidden model defaults or automatic paid calls during import/tests. The final attacker's model ID must differ from the inference model's.

To run the repo's 100-trace dataset, set your AWS credentials (and `OPENAI_API_KEY` only if you use `--web-search`), then run:

```bash
mkdir -p local_results
python examples/run_trace.py ../backend/data/source_traces.jsonl \
  --trace-id microsoft_activision_2022_t01 \
  --ground examples/ground.example.json \
  --inference-model "$INFERENCE_MODEL" \
  --anonymizer-model "$ANONYMIZER_MODEL" \
  --generator-model "$GENERATOR_MODEL" \
  --attacker-model "$ATTACKER_MODEL" \
  --abstraction-rounds 3 --outer-rounds 2 \
  --out local_results/synthetic_trace.json
```

The runner's model flags are optional: by default stages 1-2 (inference, anonymizer, generator) use GPT 6 Luna (`global.openai.gpt-6-luna`, max reasoning) and the attacker uses GPT 6 Sol (`global.openai.gpt-6-sol`, xhigh reasoning). Pass any `--*-model` flag to override one.

Add `--web-search` to run the attacker on OpenAI with web search (then `--attacker-model` is an OpenAI model ID). Use `--region` to pick a Bedrock region and `--bedrock-structured text` for models without forced tool calls.

The runner writes only a passing synthetic trace, as one line in the `source_traces.jsonl` format. It exits without writing a candidate on failure and refuses to overwrite an existing output unless requested. Optional `--rules rules.json` accepts the `WorldRules` fields in JSON form.

For N segments, the maximum per candidate is approximately `2 * N * K_abs + 2` model requests: local inference and rewriting, one world generation and one final attack. Early stopping reduces this. Web-search tool work is additional. Model/context limits may require choosing smaller trace segments; this implementation does not silently truncate inputs.

## Tests and implementation scope

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

To include the prior dataset compatibility check:

```bash
PYTHONPATH=src \
ADVERSARIAL_TRACES_DATASET=../backend/data/source_traces.jsonl \
python -m unittest discover -s tests -v
```

Tests cover current-text-only inference, exact round bounds, requested attributes, ground-truth isolation, whole-trace hints, shared filling, original input preservation, fuzzy alias matching, the real-name leak check, tool structure, coherence rules, and provider/search failures for both backends. The importer was checked against all 100 previously generated traces. Tests and examples use doubles; no live model effectiveness or privacy experiment has been run as part of building this package.

Core algorithm files are `core.py`, `matching.py` and `trace.py`; model interfaces live in `models.py`. `adapters.py` supplies prompts and output parsing. `bedrock_backend.py` and `openai_backend.py` supply the optional transports. Hidden model reasoning is neither requested nor extracted; `reasoning` fields mean short visible evidence explanations.

## References

- Staab, Vero, Balunović and Vechev, **Language Models are Advanced Anonymizers**, ICLR 2025: [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f478b2e8ad9ff0756bf5b79fb31c330f-Paper-Conference.pdf), [authors' code](https://github.com/eth-sri/llm-anonymization). The inner current-text feedback loop follows the supplied formulation of this method. Coherent trace generation and the whole-trace web attack are the requested extension.
- Official OpenAI [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and [web search](https://developers.openai.com/api/docs/guides/tools-web-search) documentation informed the optional backend.
