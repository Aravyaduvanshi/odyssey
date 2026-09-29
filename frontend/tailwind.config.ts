import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./hooks/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#111827",
        panel: "#f7f8fb",
        signal: "#0f766e",
        caution: "#b45309",
        road: "#334155"
      }
    }
  },
  plugins: []
};

export default config;

