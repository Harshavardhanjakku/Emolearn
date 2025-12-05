// Groq API Configuration
const GROQ_API_KEY = process.env.GROQ_API_KEY

if (!GROQ_API_KEY) {
  throw new Error('GROQ_API_KEY is not defined in environment variables')
}

// API Configuration
export const GROQ_CONFIG = {
  apiKey: GROQ_API_KEY,
  model: "llama-3.3-70b-versatile", // Broadly supported model
  maxTokens: 2000,
  temperature: 0.7,
  topP: 0.9,
}