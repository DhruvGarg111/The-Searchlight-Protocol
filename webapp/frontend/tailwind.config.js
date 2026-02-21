/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Orbitron", "Segoe UI", "sans-serif"],
        body: ["Space Grotesk", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        "accent-guide": "#7c5cff",
        "accent-slicer": "#22d3ee",
        "accent-detector": "#f5bd5a",
      },
    },
  },
  plugins: [],
};
