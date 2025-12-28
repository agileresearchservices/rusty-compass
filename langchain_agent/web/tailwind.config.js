/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom colors for agent observability
        'agent': {
          'query': '#3b82f6',    // blue-500 - query evaluator
          'search': '#8b5cf6',   // violet-500 - hybrid search
          'rerank': '#a855f7',   // purple-500 - reranker
          'grade': '#10b981',    // emerald-500 - document grader
          'transform': '#f59e0b', // amber-500 - query transformer
          'llm': '#06b6d4',      // cyan-500 - LLM response
          'response': '#ec4899', // pink-500 - response grader
          'success': '#22c55e',  // green-500
          'error': '#ef4444',    // red-500
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in': 'slideIn 0.3s ease-out',
      },
      keyframes: {
        slideIn: {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
