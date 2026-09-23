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

For the included provider adapter:

```bash
python -m pip install '.[openai]'
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
    GroundTruth, synthesize_trace, trace_from_dict, trace_to_dict,
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
    synthetic_record = trace_to_dict(result.trace)
else:
    assert result.trace is None
    print(result.reason)  # sanitized category, no rejected trace or answer key
```

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

The default local matcher checks every candidate. It counts either an exact normalized matter-identity alias or all required party groups appearing together in one guess, using distinct matched parties. It normalizes Unicode, capitalization, punctuation and whitespace; it does not use fuzzy substring matching or a model judge. Certainty does not discard an otherwise correct guess.

Use an answer key for the selected trace, including likely buyer/target aliases. Missing aliases or a wrong answer key can produce a false apparent pass. A custom `matcher(report, ground)` must remain local and deterministic to preserve the stated ground-truth boundary.

| Result | Meaning |
| --- | --- |
| `passed_attack` | A completed attack did not match the supplied ground truth; `trace` contains the candidate |
| `failed`, `reidentified` | The attacker kept identifying the original within the allowed rounds |
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

With `require_web_search=True`, an attack without a confirmed completed search fails as `invalid_attack`. A custom final attacker must honestly report completed web-search execution; the included provider backend derives that flag from actual completed search output, not from model-written JSON. The example runner takes `--web-search` for the same thing.

## Use the optional API adapter

```python
from adversarial_traces.adapters import (
    PromptInferenceModel, PromptAnonymizerModel,
    PromptWorldGenerator, PromptFinalAttacker,
)
from adversarial_traces.openai_backend import OpenAIResponsesBackend

inference_model = PromptInferenceModel(OpenAIResponsesBackend(inference_model_name))
anonymizer_model = PromptAnonymizerModel(OpenAIResponsesBackend(anonymizer_model_name))
generator = PromptWorldGenerator(OpenAIResponsesBackend(generator_model_name))
final_attacker = PromptFinalAttacker(OpenAIResponsesBackend(attacker_model_name))
```

Choose model IDs supported by your account: all need structured outputs, and the final model needs web search only if you turn it on. There are no hidden model defaults or automatic paid calls during import/tests. The adapter makes a fresh Responses request per call, uses `store=False`, passes no conversation or prior-response ID, and disables SDK retries for clients it creates. `store=False` does not itself promise provider zero retention.

To use the previous 100-trace dataset, extract its archive, configure `OPENAI_API_KEY` locally, and execute the runner explicitly:

```bash
mkdir -p local_results
python examples/run_trace.py ../legal-agent-traces/source_traces.jsonl \
  --trace-id microsoft_activision_2022_t01 \
  --ground examples/ground.example.json \
  --inference-model "$INFERENCE_MODEL" \
  --anonymizer-model "$ANONYMIZER_MODEL" \
  --generator-model "$GENERATOR_MODEL" \
  --attacker-model "$ATTACKER_MODEL" \
  --abstraction-rounds 3 --outer-rounds 2 \
  --out local_results/synthetic_trace.json
```

The runner writes only a passing synthetic trace. It exits without writing a candidate on failure and refuses to overwrite an existing output unless requested. Optional `--rules rules.json` accepts the `WorldRules` fields in JSON form.

For N segments, the maximum per candidate is approximately `2 * N * K_abs + 2` model requests: local inference and rewriting, one world generation and one final attack. Early stopping reduces this. Web-search tool work is additional. Model/context limits may require choosing smaller trace segments; this implementation does not silently truncate inputs.

## Tests and implementation scope

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

To include the prior dataset compatibility check:

```bash
PYTHONPATH=src \
ADVERSARIAL_TRACES_DATASET=../legal-agent-traces/source_traces.jsonl \
python -m unittest discover -s tests -v
```

Tests cover current-text-only inference, exact round bounds, requested attributes, ground-truth isolation, whole-trace hints, shared filling, original input preservation, aliases, tool structure, coherence rules, and provider/search failures. The importer was checked against all 100 previously generated traces. Tests and examples use doubles; no live model effectiveness or privacy experiment has been run as part of building this package.

Core algorithm files are `core.py`, `matching.py` and `trace.py`; model interfaces live in `models.py`. `adapters.py` supplies prompts and output parsing, and `openai_backend.py` supplies the optional transport. Hidden model reasoning is neither requested nor extracted; `reasoning` fields mean short visible evidence explanations.

## References

- Staab, Vero, Balunović and Vechev, **Language Models are Advanced Anonymizers**, ICLR 2025: [paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f478b2e8ad9ff0756bf5b79fb31c330f-Paper-Conference.pdf), [authors' code](https://github.com/eth-sri/llm-anonymization). The inner current-text feedback loop follows the supplied formulation of this method. Coherent trace generation and the whole-trace web attack are the requested extension.
- Official OpenAI [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and [web search](https://developers.openai.com/api/docs/guides/tools-web-search) documentation informed the optional backend.
