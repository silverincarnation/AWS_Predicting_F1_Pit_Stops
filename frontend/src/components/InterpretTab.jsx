import { useState, useEffect } from "react";
import { api } from "../api.js";

// ── Horizontal importance bar ──────────────────────────────────────────────
function ImportanceBar({ share, color = "#3b82f6" }) {
  return (
    <div style={{ flex: 1, background: "#0f1117", borderRadius: 4, height: 10, overflow: "hidden" }}>
      <div style={{ width: `${(share * 100).toFixed(1)}%`, background: color, height: "100%", borderRadius: 4, transition: "width 0.4s" }} />
    </div>
  );
}

// ── p-value badge ──────────────────────────────────────────────────────────
function PBadge({ p }) {
  const sig = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns";
  const style = {
    display: "inline-block", borderRadius: 4, fontSize: "0.7rem", fontWeight: 700,
    padding: "2px 7px",
    background: p < 0.05 ? "#064e3b" : "#1f2937",
    color:      p < 0.05 ? "#34d399" : "#6b7280",
  };
  return <span style={style}>{sig} {p.toFixed(4)}</span>;
}

// ── LightGBM tab ───────────────────────────────────────────────────────────
function LGBMInterpret() {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    api.interpretLgbm().then(setData).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading feature importances…</div>;
  if (error)   return <div className="error-msg">⚠ {error}</div>;

  const maxImp = data.features[0]?.importance || 1;

  return (
    <div className="card">
      <h2>LightGBM — Feature Importance (split count)</h2>
      <p style={{ color: "#64748b", fontSize: "0.8rem", marginBottom: 20 }}>
        Split count: how many times each feature was used to split a tree node across all {data.features.length} features.
        Higher = more influential in the model's decisions.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {data.features.map((f, i) => (
          <div key={f.feature} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 24, color: "#64748b", fontSize: "0.75rem", textAlign: "right" }}>{i + 1}</div>
            <div style={{ width: 210, fontSize: "0.85rem", color: "#cbd5e1", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {f.label}
            </div>
            <ImportanceBar share={f.importance / maxImp} color="#3b82f6" />
            <div style={{ width: 60, textAlign: "right", fontSize: "0.8rem", color: "#93c5fd", fontWeight: 600 }}>
              {f.importance.toLocaleString()}
            </div>
            <div style={{ width: 50, textAlign: "right", fontSize: "0.75rem", color: "#64748b" }}>
              {(f.share * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Logistic Regression tab ────────────────────────────────────────────────
function LRInterpret() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    api.interpretLr().then(setData).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading">Loading LR coefficients…</div>;
  if (error)   return <div className="error-msg">⚠ {error}</div>;

  const maxAbs = Math.max(...data.features.map(f => Math.abs(f.coef)));

  return (
    <div className="card">
      <h2>Logistic Regression — Coefficients & Significance</h2>
      <p style={{ color: "#64748b", fontSize: "0.8rem", marginBottom: 4 }}>
        Intercept: <strong style={{ color: "#cbd5e1" }}>{data.intercept}</strong>.
        Features are standardised (mean=0, std=1) so coefficients are directly comparable.
        Positive coef → increases pit probability; negative → decreases it.
      </p>
      <p style={{ color: "#64748b", fontSize: "0.75rem", marginBottom: 20 }}>
        p-values via Wald test (approximate — L2 regularisation with C={" "}
        <strong style={{ color: "#cbd5e1" }}>1.0</strong>). *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns not significant.
      </p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #2d3748", color: "#64748b", fontSize: "0.75rem", textTransform: "uppercase" }}>
              <th style={{ padding: "8px 12px", textAlign: "left" }}>Feature</th>
              <th style={{ padding: "8px 12px", textAlign: "center" }}>Direction</th>
              <th style={{ padding: "8px 12px", textAlign: "center" }}>Coefficient</th>
              <th style={{ padding: "8px 12px", textAlign: "center" }}>Odds Ratio</th>
              <th style={{ padding: "8px 12px", textAlign: "center" }}>z-stat</th>
              <th style={{ padding: "8px 12px", textAlign: "center" }}>p-value</th>
              <th style={{ padding: "8px 8px", minWidth: 120 }}>Magnitude</th>
            </tr>
          </thead>
          <tbody>
            {data.features.map((f, i) => {
              const pos = f.coef >= 0;
              return (
                <tr key={f.feature} style={{ borderBottom: "1px solid #1e293b", background: i % 2 === 0 ? "transparent" : "#0d111a" }}>
                  <td style={{ padding: "10px 12px", color: "#cbd5e1", fontWeight: 500 }}>
                    <div>{f.label}</div>
                    <div style={{ fontSize: "0.7rem", color: "#475569" }}>{f.feature}</div>
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <span style={{ color: pos ? "#34d399" : "#f87171", fontSize: "1.1rem" }}>{pos ? "▲" : "▼"}</span>
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "center", fontFamily: "monospace", color: pos ? "#34d399" : "#f87171", fontWeight: 600 }}>
                    {f.coef > 0 ? "+" : ""}{f.coef.toFixed(4)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "center", color: "#e2e8f0" }}>
                    {f.odds_ratio.toFixed(3)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "center", color: "#94a3b8", fontFamily: "monospace" }}>
                    {f.z_stat.toFixed(2)}
                  </td>
                  <td style={{ padding: "10px 12px", textAlign: "center" }}>
                    <PBadge p={f.p_value} />
                  </td>
                  <td style={{ padding: "10px 8px" }}>
                    <div style={{ background: "#0f1117", borderRadius: 4, height: 8, overflow: "hidden" }}>
                      <div style={{
                        width: `${(Math.abs(f.coef) / maxAbs * 100).toFixed(1)}%`,
                        background: pos ? "#059669" : "#dc2626",
                        height: "100%", borderRadius: 4, transition: "width 0.4s",
                      }} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main tab ───────────────────────────────────────────────────────────────
export default function InterpretTab() {
  const [modelKey, setModelKey] = useState("lgbm");

  return (
    <div>
      <div className="card">
        <h2>Model Interpretability</h2>
        <p style={{ color: "#64748b", fontSize: "0.875rem", lineHeight: 1.6 }}>
          Extracted directly from the loaded models at runtime — no additional training data needed.
          LightGBM uses split-count feature importance; Logistic Regression uses standardised coefficients
          with Wald-test significance.
        </p>
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
        {[
          { id: "lgbm", label: "LightGBM — Feature Importance" },
          { id: "lr",   label: "Logistic Regression — Coefficients" },
        ].map(m => (
          <button
            key={m.id}
            className={`ale-tab-btn ${modelKey === m.id ? "active" : ""}`}
            onClick={() => setModelKey(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {modelKey === "lgbm" ? <LGBMInterpret /> : <LRInterpret />}
    </div>
  );
}
