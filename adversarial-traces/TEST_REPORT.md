# Verification report

- Python: 3.12.14 (rechecked on 3.11.15 after making web search optional: 82 tests passed).
- 79 tests passed against the source package.
- The same 79 tests passed against the built and separately installed wheel (the wheel is not checked in; rebuild it with the command below).
- The optional dataset integration test imported and validated all 100 prior source traces.
- The offline example completed two scripted outer rounds: reidentified, then passed_attack; one world was generated per candidate.
- The real-runner command was checked without model calls, including zero-budget failure, mocked success/failure, trace-ID selection, output refusal/overwrite and private file permissions.
- No paid API call, live model evaluation, live web-search attack, or differential-privacy mechanism was run.

Build command used:

```bash
python -m pip wheel . --no-deps --no-build-isolation --no-index --wheel-dir dist
```

Test command (set the dataset path when available):

```bash
PYTHONPATH=src ADVERSARIAL_TRACES_DATASET=../legal-agent-traces/source_traces.jsonl python -m unittest discover -s tests
```

The tests verify contracts, control flow, structural/coherence checks and failure handling. They do not establish anonymization efficacy, legal accuracy or provider retention guarantees.
