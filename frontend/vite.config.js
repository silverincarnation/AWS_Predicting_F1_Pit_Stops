import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API calls to FastAPI during local development
    proxy: {
      "/api":      { target: "http://localhost:8080", changeOrigin: true },
      "/predict":  { target: "http://localhost:8080", changeOrigin: true },
      "/health":   { target: "http://localhost:8080", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
  },
});
