/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#10141A",
        surface: "#171C24",
        border: "#262D38",
        ink: "#E7E5DF",
        muted: "#8B93A1",
        accent: "#E8A33D",
        masked: "#1D2430",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
