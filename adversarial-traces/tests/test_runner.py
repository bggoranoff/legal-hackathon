"""Example runner tests: several traces in parallel, with synthesis faked (no API calls)."""
import importlib.util
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from adversarial_traces import trace_from_dict
from adversarial_traces.models import AttemptSummary, SynthesisResult

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_trace", ROOT / "examples" / "run_trace.py")
run_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_trace)


def record(trace_id, workflow="structure_payment"):
    return {"trace_id": trace_id, "workflow": workflow, "events": [
        {"event_id": f"{trace_id}.e1", "sequence": 1, "role": "user", "event_type": "message",
         "content": f"Review {trace_id}."},
    ]}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.input = Path(self.dir.name) / "in.jsonl"
        self.input.write_text("".join(json.dumps(record(f"t{i}")) + "\n" for i in range(3)))
        self.out = Path(self.dir.name) / "out.jsonl"
        self.threads = set()

    def fake_synthesize(self, fail=()):
        def synthesize(trace, ground, *args, **kwargs):
            self.threads.add(threading.get_ident())
            time.sleep(0.05)
            text = trace.segments[0].payload["text"]
            if any(f"Review {name}." == text for name in fail):
                return SynthesisResult("failed", None, (AttemptSummary(1, "reidentified"),), "reidentified")
            fake = trace_from_dict({"trace_id": "x", "events": [
                {"event_id": "e1", "sequence": 1, "role": "user", "event_type": "message",
                 "content": text.replace("Review", "Synthetic")}]})
            return SynthesisResult("passed_attack", fake, (AttemptSummary(1, "passed_attack"),))
        return synthesize

    def run_main(self, *extra, fail=()):
        err = io.StringIO()
        with mock.patch.object(run_trace, "synthesize_trace", self.fake_synthesize(fail)), redirect_stderr(err):
            code = run_trace.main([str(self.input), "--out", str(self.out), *extra])
        return code, err.getvalue()

    def test_two_traces_convert_in_parallel_in_selected_order(self):
        code, err = self.run_main("--trace-id", "t2", "--trace-id", "t0")
        self.assertEqual(0, code)
        lines = [json.loads(line) for line in self.out.read_text().splitlines()]
        self.assertEqual(["Synthetic t2.", "Synthetic t0."], [r["events"][0]["content"] for r in lines])
        self.assertEqual(2, len(self.threads))
        self.assertIn('"passed": 2', err)

    def test_first_n_and_partial_failure_writes_passes_only(self):
        code, err = self.run_main("--first", "3", fail=("t1",))
        self.assertEqual(1, code)
        lines = [json.loads(line) for line in self.out.read_text().splitlines()]
        self.assertEqual(["Synthetic t0.", "Synthetic t2."], [r["events"][0]["content"] for r in lines])
        self.assertIn('"trace": "t1", "status": "failed"', err)

    def test_answer_key_with_several_traces_is_refused(self):
        ground = Path(self.dir.name) / "ground.json"
        ground.write_text(json.dumps({"identities": ["Some deal"]}))
        code, err = self.run_main("--first", "2", "--ground", str(ground))
        self.assertEqual(2, code)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
