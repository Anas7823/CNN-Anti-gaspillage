/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          50:  '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
          950: '#052e16',
        },
        cream: '#fefce8',
        rotten: {
          light: '#fed7aa',
          DEFAULT: '#f97316',
          dark: '#c2410c',
        },
        fresh: {
          light: '#bbf7d0',
          DEFAULT: '#22c55e',
          dark: '#15803d',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'system-ui', 'sans-serif'],
      },
      animation: {
        'scan': 'scan 2s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 1.5s ease-in-out infinite',
        'fade-up': 'fade-up 0.6s ease-out',
        'slide-in': 'slide-in 0.4s ease-out',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { transform: 'translateY(-100%)', opacity: '0' },
          '50%':       { transform: 'translateY(0%)',    opacity: '1' },
        },
        'pulse-ring': {
          '0%':   { transform: 'scale(0.8)',  opacity: '0.8' },
          '100%': { transform: 'scale(1.4)',  opacity: '0' },
        },
        'fade-up': {
          '0%':   { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in': {
          '0%':   { opacity: '0', transform: 'translateX(-20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':       { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
