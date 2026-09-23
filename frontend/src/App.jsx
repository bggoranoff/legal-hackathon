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
  return text
    .split("\u0001")
    .map((part, i) => (i % 2 === 1 ? <mark className="pii" key={i}>{part}</mark> : part));
}

const STEP_LIMIT = 240;

function clip(text) {
  if (text.length <= STEP_LIMIT) return { text, clipped: false };
  let s = text.slice(0, STEP_LIMIT);
  if ((s.split("\u0001").length - 1) % 2 === 1) s += "\u0001"; // close a bold span cut mid-way
  return { text: s + " …", clipped: true };
}

function Modal({ trace, step, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title">
            {trace.id} — Step {step.step}: {step.action}
          </div>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="modal-detail">{detailParts(step.detail)}</div>
      </div>
    </div>
  );
}

function scoreClass(score) {
  if (score >= 50) return "bad";
  if (score >= 10) return "warn";
  return "good";
}

function dotClass(score) {
  if (score > 90) return "red"; // deal re-identified
  if (score === 0) return "green"; // nothing recovered
  return "yellow"; // partially recovered
}

function TraceView({ trace, onOpenStep }) {
  if (!trace) return <div className="empty">Select a trace.</div>;
  return (
    <div className="trace">
      <div className="trace-scroll">
      <div className="trace-title">
        {trace.id} — {trace.matter}
      </div>
      {trace.steps.map((s) => {
        const { text, clipped } = clip(s.detail);
        return (
          <div
            className={"step" + (clipped ? " clickable" : "")}
            key={s.step}
            onClick={clipped ? () => onOpenStep(trace, s) : undefined}
          >
            <div className="step-head">
              Step {s.step}: {s.action}
            </div>
            <div className="step-detail">{detailParts(text)}</div>
          </div>
        );
      })}
      </div>
    </div>
  );
}

function TraceList({ traces, selected, onSelect, onOpenStep }) {
  // Selection is shared across lists by row position, so the same matter's original,
  // transformed and reconstructed variants line up when shown side by side.
  return (
    <div className="columns">
      <ul className="list">
        {traces.map((t, i) => (
          <li key={t.id}>
            <button
              className={"row" + (i === selected ? " active" : "")}
              onClick={() => onSelect(i)}
            >
              <span className="row-id">
                {t.score != null && <span className={"dot " + dotClass(t.score)} />}
                {t.id}
              </span>
              <span className="row-matter">{t.matter}</span>
            </button>
          </li>
        ))}
      </ul>
      <TraceView trace={selected == null ? null : traces[selected]} onOpenStep={onOpenStep} />
    </div>
  );
}

export default function App() {
  const [originals, setOriginals] = useState([]);
  const [selected, setSelected] = useState(null); // shared row index across all lists

  const [mode, setMode] = useState(null);
  const [generated, setGenerated] = useState(null);
  const [reconstructed, setReconstructed] = useState(null);

  const [score, setScore] = useState(null);
  const [modalStep, setModalStep] = useState(null);

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
    setReconstructed(null);
    setScore(null);
  }

  async function reconstruct() {
    const traces = await postJson("/api/reconstruct", { mode });
    setReconstructed(traces);
    setScore(null);
  }

  async function verify() {
    const { score } = await postJson("/api/verify", { mode });
    setScore(score);
  }

  const openStep = (trace, step) => setModalStep({ trace, step });

  return (
    <div className="app">
      <h1>Orizon</h1>
      <p className="tagline">
        A clean room for AI agent logs in regulated industries (e.g. legal).
      </p>

      <div className="section">
        <h2>Original traces</h2>
        <TraceList
          traces={originals}
          selected={selected}
          onSelect={setSelected}
          onOpenStep={openStep}
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
            selected={selected}
            onSelect={setSelected}
            onOpenStep={openStep}
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
            selected={selected}
            onSelect={setSelected}
            onOpenStep={openStep}
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
              {(() => {
                const sel = reconstructed && selected != null ? reconstructed[selected] : null;
                const big = sel && sel.score != null ? sel.score : score;
                return (
                  <>
                    <div className={"num " + scoreClass(big)}>{big}%</div>
                    <div>
                      of the deal identity recovered from{" "}
                      {sel ? sel.id : `the ${mode} traces`}.
                    </div>
                    <div className="score-avg">
                      Average: {score}% across {reconstructed.length} {mode} traces
                    </div>
                  </>
                );
              })()}
            </div>
          ) : (
            <div className="empty">Verify how much PII the reconstruction recovered.</div>
          )}
        </div>
      </div>

      {modalStep && (
        <Modal trace={modalStep.trace} step={modalStep.step} onClose={() => setModalStep(null)} />
      )}
    </div>
  );
}
