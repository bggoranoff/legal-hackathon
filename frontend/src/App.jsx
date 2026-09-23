import { useEffect, useState } from "react";

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

function detailParts(text) {
  // Backend wraps each PII span in \u0001 sentinels; odd segments are the PII.
  return text.split("\u0001").map((part, i) => (i % 2 === 1 ? <b key={i}>{part}</b> : part));
}

function scoreClass(score) {
  if (score >= 50) return "bad";
  if (score >= 10) return "warn";
  return "good";
}

function TraceView({ trace }) {
  if (!trace) return <div className="empty">Select a trace.</div>;
  return (
    <div className="trace">
      <div className="trace-title">
        {trace.id} — {trace.matter}
      </div>
      {trace.steps.map((s) => (
        <div className="step" key={s.step}>
          <div className="step-head">
            Step {s.step}: {s.action}
          </div>
          <div className="step-detail">{detailParts(s.detail)}</div>
        </div>
      ))}
    </div>
  );
}

function TraceList({ traces, selectedId, onSelect }) {
  return (
    <div className="columns">
      <ul className="list">
        {traces.map((t) => (
          <li key={t.id}>
            <button
              className={"row" + (t.id === selectedId ? " active" : "")}
              onClick={() => onSelect(t.id)}
            >
              <span className="row-id">{t.id}</span>
              <span className="row-matter">{t.matter}</span>
            </button>
          </li>
        ))}
      </ul>
      <TraceView trace={traces.find((t) => t.id === selectedId)} />
    </div>
  );
}

export default function App() {
  const [originals, setOriginals] = useState([]);
  const [originalSel, setOriginalSel] = useState(null);

  const [mode, setMode] = useState(null);
  const [generated, setGenerated] = useState(null);
  const [generatedSel, setGeneratedSel] = useState(null);

  const [reconstructed, setReconstructed] = useState(null);
  const [reconstructedSel, setReconstructedSel] = useState(null);

  const [score, setScore] = useState(null);

  useEffect(() => {
    fetch("/api/traces/original")
      .then((r) => r.json())
      .then(setOriginals);
  }, []);

  async function transform(nextMode) {
    const url = nextMode === "redact" ? "/api/redact" : "/api/synthetic";
    const traces = await postJson(url);
    setMode(nextMode);
    setGenerated(traces);
    setGeneratedSel(null);
    setReconstructed(null);
    setReconstructedSel(null);
    setScore(null);
  }

  async function reconstruct() {
    const traces = await postJson("/api/reconstruct", { mode });
    setReconstructed(traces);
    setReconstructedSel(null);
    setScore(null);
  }

  async function verify() {
    const { score } = await postJson("/api/verify", { mode });
    setScore(score);
  }

  return (
    <div className="app">
      <h1>Orizon</h1>
      <p className="tagline">
        A clean room for agent traces in regulated industries.
      </p>

      <div className="section">
        <h2>Original traces</h2>
        <TraceList
          traces={originals}
          selectedId={originalSel}
          onSelect={setOriginalSel}
        />
      </div>

      <div className="section">
        <h2>Transform</h2>
        <div className="btn-row">
          <button className="btn-redact" onClick={() => transform("redact")}>
            Redact
          </button>
          <button className="btn-synthetic" onClick={() => transform("synthetic")}>
            Synthetic
          </button>
        </div>
      </div>

      <div className="section">
        <h2>Generated traces{mode ? ` — ${mode}` : ""}</h2>
        {generated ? (
          <TraceList
            traces={generated}
            selectedId={generatedSel}
            onSelect={setGeneratedSel}
          />
        ) : (
          <div className="empty">Run Redact or Synthetic above.</div>
        )}
      </div>

      <div className="section">
        <h2>Reconstruct</h2>
        <div className="btn-row">
          <button className="btn-reconstruct" onClick={reconstruct} disabled={!generated}>
            Reconstruct
          </button>
        </div>
        {reconstructed ? (
          <TraceList
            traces={reconstructed}
            selectedId={reconstructedSel}
            onSelect={setReconstructedSel}
          />
        ) : (
          <div className="empty">Reconstruct from the generated traces above.</div>
        )}
      </div>

      <div className="section">
        <h2>Verify</h2>
        <div className="btn-row">
          <button className="btn-verify" onClick={verify} disabled={!reconstructed}>
            Verify
          </button>
        </div>
        <div className="verify-pane">
          {score !== null ? (
            <div className="score">
              <div className={"num " + scoreClass(score)}>{score}%</div>
              <div>of original PII recovered from the {mode} traces.</div>
            </div>
          ) : (
            <div className="empty">Verify how much PII the reconstruction recovered.</div>
          )}
        </div>
      </div>
    </div>
  );
}
