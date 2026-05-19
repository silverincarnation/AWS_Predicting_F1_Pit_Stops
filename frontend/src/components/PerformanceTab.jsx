import { useState, useEffect } from "react";
import { api } from "../api.js";

function MetricBox({ value, label, highlight }) {
  return (
    <div className={`metric-box ${highlight ? "highlight" : ""}`}>
      <div className="m-value">{typeof value === "number" ? value.toFixed(4) : value}</div>
      <div className="m-label">{label}</div>
    </div>
  );
}

function ModelMetrics({ name, m, highlightKey }) {
  const keys = ["roc_auc", "f1", "precision", "recall", "accuracy"];
  const labels = { roc_auc: "ROC AUC", f1: "F1", precision: "Precision", recall: "Recall", accuracy: "Accuracy" };
  return (
    <div className="card">
      <h2>{name}</h2>
      <div style={{ marginBottom: 12 }}>
        <span className="badge badge-green" style={{ marginRight: 8 }}>
          CV AUC {m.cv_auc?.toFixed(4)} ± {m.cv_auc_std?.toFixed(4)}
        </span>
        {m.best_iteration &&
          <span className="badge" style={{ background: "#1e3a5f", color: "#93c5fd" }}>
            Best iter {m.best_iteration}
          </span>}
      </div>
      <div className="metrics-row">
        {keys.map(k => (
          <MetricBox key={k} value={m[k]} label={labels[k]} highlight={k === highlightKey} />
        ))}
      </div>
    </div>
  );
}

export default function PerformanceTab() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    api.metrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading performance data…</div>;
  if (error)   return <div className="error-msg">⚠ {error}</div>;

  const winner = metrics.winner_by_roc_auc;
  const lgbm   = metrics.lightgbm;
  const lr     = metrics.logistic_regression;

  return (
    <div>
      <div className="card">
        <h2>Model Comparison</h2>
        <p style={{ color: "#64748b", fontSize: "0.875rem" }}>
          Both models trained with 5-fold stratified CV. Best model by holdout ROC-AUC:{" "}
          <strong style={{ color: "#34d399" }}>{winner}</strong>.
          Run timestamp: {metrics.run_timestamp}.
        </p>
      </div>

      <ModelMetrics name="LightGBM"            m={lgbm} highlightKey="roc_auc" />
      <ModelMetrics name="Logistic Regression" m={lr}   highlightKey="roc_auc" />

    </div>
  );
}
