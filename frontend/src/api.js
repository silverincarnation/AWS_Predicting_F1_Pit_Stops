// Base URL: use VITE_API_URL env var in production (points to ALB).
// In development, Vite's proxy forwards /api and /predict to localhost:8080.
const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  health:         ()           => request("/health"),
  features:       ()           => request("/api/features"),
  metrics:        ()           => request("/api/metrics"),
  plotUrl:        (name)       => request(`/api/plots/${name}`),
  interpretLgbm:  ()           => request("/api/interpret/lgbm"),
  interpretLr:    ()           => request("/api/interpret/lr"),
  predict:        (body, model = "lgbm") =>
    request(`/predict?model=${model}`, { method: "POST", body: JSON.stringify(body) }),
  predictCompare: (body)       =>
    request("/predict/compare",        { method: "POST", body: JSON.stringify(body) }),
};
