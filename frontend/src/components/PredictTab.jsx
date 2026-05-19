import { useState } from "react";
import { api } from "../api.js";

const FIELDS = [
  { key: "TyreLife",                 label: "Tyre Life (laps)",        step: 1,    min: 0 },
  { key: "Cumulative_Degradation",   label: "Cumulative Degradation",  step: 0.01, min: 0 },
  { key: "LapNumber",                label: "Lap Number",              step: 1,    min: 1 },
  { key: "RaceProgress",             label: "Race Progress (0–1)",     step: 0.01, min: 0, max: 1 },
  { key: "LapTime_Delta",            label: "Lap Time Delta (s)",      step: 0.1 },
  { key: "Stint",                    label: "Stint",                   step: 1,    min: 1 },
  { key: "Position",                 label: "Position",                step: 1,    min: 1, max: 20 },
  { key: "TyreLife_LapNumber_ratio", label: "TyreLife / LapNumber",   step: 0.001, min: 0 },
  { key: "Compound_te",              label: "Compound Target Enc.",    step: 0.01, min: 0, max: 1 },
  { key: "Race_te",                  label: "Circuit Target Enc.",     step: 0.01, min: 0, max: 1 },
];

const DEFAULTS = {
  TyreLife: 22, Cumulative_Degradation: 0.38, LapNumber: 34,
  RaceProgress: 0.61, LapTime_Delta: 0.4, Stint: 1, Position: 5,
  TyreLife_LapNumber_ratio: 0.647, Compound_te: 0.21, Race_te: 0.19,
};

function ProbBar({ prob }) {
  const pct = Math.round(prob * 100);
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-bg">
        <div
          className={`prob-bar-fill ${prob >= 0.5 ? "high" : "low"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="prob-bar-labels"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
  );
}

function ResultCard({ result, title }) {
  if (!result) return null;
  const pit = result.pit_next_lap;
  return (
    <div className={`result-card ${pit ? "result-pit" : "result-no-pit"}`}>
      {title && <div style={{ fontSize: "0.75rem", color: "#64748b", marginBottom: 8 }}>{title}</div>}
      <div className="result-label">{pit ? "PIT" : "STAY OUT"}</div>
      <div className="result-prob">
        Probability: <span>{(result.probability * 100).toFixed(1)}%</span>
        {" "}· Threshold: <span>{(result.threshold * 100).toFixed(0)}%</span>
        {" "}· Model: <span>{result.model_used}</span>
      </div>
      <ProbBar prob={result.probability} />
    </div>
  );
}

export default function PredictTab() {
  const [form, setForm]       = useState(DEFAULTS);
  const [mode, setMode]       = useState("lgbm"); // "lgbm" | "lr"
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleChange = (key, val) =>
    setForm(f => ({ ...f, [key]: val === "" ? "" : parseFloat(val) }));

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (mode === "compare") {
        setResult({ compare: await api.predictCompare(form) });
      } else {
        setResult({ single: await api.predict(form, mode) });
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2>Choose Model</h2>
        <div className="model-selector">
          {[
            { id: "lgbm",    name: "LightGBM",            auc: "AUC 0.9433 · F1 0.73 · Recommended" },
            { id: "lr",      name: "Logistic Regression",  auc: "AUC 0.8449 · Baseline" },
          ].map(m => (
            <button
              key={m.id}
              className={`model-btn ${mode === m.id ? "active" : ""}`}
              onClick={() => setMode(m.id)}
            >
              <span className="model-name">{m.name}</span>
              <span className="model-auc">{m.auc}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Input Features</h2>
        <div className="form-grid">
          {FIELDS.map(f => (
            <div className="form-field" key={f.key}>
              <label>{f.label}</label>
              <input
                type="number"
                step={f.step}
                min={f.min}
                max={f.max}
                value={form[f.key]}
                onChange={e => handleChange(f.key, e.target.value)}
              />
            </div>
          ))}
        </div>
        <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
          {loading ? "Predicting…" : "Predict"}
        </button>
        {error && <div className="error-msg">⚠ {error}</div>}
      </div>

      {result?.single && <ResultCard result={result.single} />}
      {result?.compare && (
        <div className="compare-grid">
          <ResultCard result={result.compare.lgbm} title="LightGBM" />
          <ResultCard result={result.compare.lr}   title="Logistic Regression" />
        </div>
      )}
      {result?.compare && (
        <div className="card" style={{ marginTop: 16, textAlign: "center" }}>
          {result.compare.models_agree
            ? <span className="badge badge-green">✓ Both models agree</span>
            : <span className="badge badge-red">⚠ Models disagree</span>}
        </div>
      )}
    </div>
  );
}
