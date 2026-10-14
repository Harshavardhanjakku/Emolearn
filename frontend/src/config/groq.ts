<<<<<<< HEAD
// Groq API Configuration (Vite injects `import.meta.env` at build time)
export const GROQ_API_KEY = (import.meta.env.VITE_GROQ_API_KEY || undefined) as string | undefined
=======
// Groq API Configuration
// In Vite, only env vars prefixed with VITE_ are exposed to the browser.
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY

if (!GROQ_API_KEY) {
  throw new Error('VITE_GROQ_API_KEY is not defined. Add it to your .env and restart the dev server.')
}
>>>>>>> 3b6ca98 (added codes)

// API Configuration
export const GROQ_CONFIG = {
  apiKey: GROQ_API_KEY,
  model: "llama-3.3-70b-versatile", // Broadly supported model
  maxTokens: 2000,
  temperature: 0.7,
  topP: 0.9,
}