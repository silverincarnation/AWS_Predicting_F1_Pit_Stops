import { useState } from "react";
import PredictTab    from "./components/PredictTab.jsx";
import VariablesTab  from "./components/VariablesTab.jsx";
import PerformanceTab from "./components/PerformanceTab.jsx";
import InterpretTab  from "./components/InterpretTab.jsx";

const TABS = [
  { id: "predict",     label: "Predict" },
  { id: "variables",   label: "Variables" },
  { id: "performance", label: "Performance" },
  { id: "interpret",   label: "Interpret" },
];

export default function App() {
  const [tab, setTab] = useState("predict");

  return (
    <div>
      <header className="app-header">
        <div>
          <h1>F1 Pit Stop Predictor</h1>
          <p>LightGBM + Logistic Regression</p>
        </div>
        <a
          href="https://www.kaggle.com/competitions/playground-series-s6e5"
          target="_blank"
          rel="noopener noreferrer"
          className="header-badge"
          style={{ textDecoration: "none" }}
        >
          Kaggle S6E5 ↗
        </a>
      </header>

      <nav className="tab-nav">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="tab-content">
        {tab === "predict"     && <PredictTab />}
        {tab === "variables"   && <VariablesTab />}
        {tab === "performance" && <PerformanceTab />}
        {tab === "interpret"   && <InterpretTab />}
      </main>
    </div>
  );
}
