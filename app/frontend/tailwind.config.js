/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        cyber: {
          bg: "#090d16",
          card: "#0f172a",
          surface: "#141e33",
          border: "#1e293b",
          primary: "#6366f1",
          secondary: "#06b6d4",
          accent: "#ec4899",
          glow: "#3b82f6",
        }
      },
      boxShadow: {
        glow: "0 0 25px -5px rgba(99, 102, 241, 0.25)",
        cyanGlow: "0 0 25px -5px rgba(6, 182, 212, 0.25)",
      }
    },
  },
  plugins: [],
}
