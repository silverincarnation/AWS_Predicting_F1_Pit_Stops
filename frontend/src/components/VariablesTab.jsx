import { useState, useEffect } from "react";
import { api } from "../api.js";

export default function VariablesTab() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    api.features()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading feature definitions…</div>;
  if (error)   return <div className="error-msg">⚠ {error}</div>;

  const { feature_cols, features } = data;

  return (
    <div>
      <div className="card">
        <h2>Feature Reference</h2>
        <p style={{ color: "#64748b", fontSize: "0.875rem", marginBottom: 0 }}>
          The model uses {feature_cols.length} features. All are required for prediction.
          Target-encoding features (Compound_te, Race_te) are precomputed averages from training data.
        </p>
      </div>

      <div className="feature-grid">
        {feature_cols.map(key => {
          const f = features[key];
          return (
            <div className="feature-card" key={key}>
              <div className="feat-name">{f.label}</div>
              <div className="feat-unit">{f.unit}</div>
              <div className="feat-desc">{f.description}</div>
              <div className="feat-meta">
                <span>Range: {f.range}</span>
                <span>Example: {f.example}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
