// Groq API Configuration (Vite injects `import.meta.env` at build time)
export const GROQ_API_KEY = (import.meta.env.VITE_GROQ_API_KEY || undefined) as string | undefined

// API Configuration
export const GROQ_CONFIG = {
  apiKey: GROQ_API_KEY,
  model: "llama-3.3-70b-versatile", // Broadly supported model
  maxTokens: 2000,
  temperature: 0.7,
  topP: 0.9,
}
