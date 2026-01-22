module.exports = {
  darkMode: 'class',
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0B0C15',
        paper: '#13141F',
        subtle: '#1C1E2D',
        foreground: {
          DEFAULT: '#E2E8F0',
          muted: '#94A3B8',
          inverse: '#0F172A',
        },
        primary: {
          DEFAULT: '#6366F1',
          hover: '#4F46E5',
          foreground: '#FFFFFF',
        },
        secondary: {
          DEFAULT: '#1E293B',
          foreground: '#F8FAFC',
        },
        accent: {
          cyan: '#06B6D4',
          purple: '#8B5CF6',
          green: '#10B981',
          pink: '#EC4899',
          orange: '#F97316',
        },
        border: {
          DEFAULT: '#2D3042',
          active: '#4F46E5',
        },
        card: {
          DEFAULT: '#13141F',
          foreground: '#E2E8F0',
        },
        destructive: {
          DEFAULT: '#EF4444',
          foreground: '#FFFFFF',
        },
        muted: {
          DEFAULT: '#1C1E2D',
          foreground: '#94A3B8',
        },
        input: '#2D3042',
        ring: '#4F46E5',
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'sans-serif'],
        heading: ['Manrope', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        lg: '0.5rem',
        md: '0.375rem',
        sm: '0.25rem',
      },
      boxShadow: {
        glow: '0 0 15px rgba(99, 102, 241, 0.3)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};