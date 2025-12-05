// groqGPT.js

const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Keep secure

export async function askGroq(topic, mode = 'simple') {
  const callId = Math.random().toString(36).slice(2, 8)
  const startedAt = Date.now()
  console.log('[askGroq] start', { callId, mode, topicPreview: String(topic).slice(0, 120), time: new Date(startedAt).toISOString() })
  let prompt;

  switch (mode) {
    case 'im5':
      prompt = `Explain the topic "${topic}" in very simple terms, like you're explaining it to a 5-year-old. Include practical real-world examples.`;
      break;
    case 'story':
      prompt = `Tell a short, engaging story that explains the topic "${topic}" in a relatable way. Make it fun and easy to understand.`;
      break;
    case 'quiz':
      prompt = `Generate a short quiz (3-5 questions) for the topic "${topic}". Include multiple choice options and the correct answers.`;
      break;
    case 'resources':
      prompt = `Provide curated study resources for "${topic}", such as PDF guides, roadmaps, YouTube video links, and helpful articles. Group them by Beginner, Intermediate, and Advanced.`;
      break;
    case 'youtube':
      // Check if the topic looks like a YouTube URL
      if (topic.includes('youtube.com') || topic.includes('youtu.be')) {
        // Import and use the YouTube summarizer
        try {
          console.log('[askGroq] youtube mode detected', { callId, topic })
          const { summarizeYouTubeVideo } = await import('./youtubeSummarizer.js');
          const ytStart = Date.now()
          const result = await summarizeYouTubeVideo(topic, 'summary');
          console.log('[askGroq] youtube summarize success', { callId, durationMs: Date.now() - ytStart, resultPreview: String(result).slice(0, 120) })
          return result;
        } catch (ytErr) {
          console.error('[askGroq] youtube summarize error', { callId, error: ytErr instanceof Error ? { name: ytErr.name, message: ytErr.message, stack: ytErr.stack } : ytErr })
          return 'Sorry, failed to summarize the YouTube video.'
        }
      } else {
        prompt = `I notice you mentioned YouTube. Please provide a YouTube video URL that you'd like me to summarize. I can create summaries, study notes, or quizzes from educational videos.`;
      }
      break;
    default:
      prompt = `Explain the topic "${topic}" in under 140 words for a beginner. Use simple language and examples.`;
  }

  try {
    const requestBody = {
      model: 'llama-3.3-70b-versatile',
      messages: [
        {
          role: 'system',
          content: 'You are a friendly and helpful AI tutor that adapts to different learning needs.',
        },
        { role: 'user', content: prompt },
      ],
    }
    console.log('[askGroq] request', { callId, endpoint: 'https://api.groq.com/openai/v1/chat/completions', bodyPreview: JSON.stringify(requestBody).slice(0, 200) })

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    const text = await response.text();
    let data
    try {
      data = JSON.parse(text)
    } catch (parseErr) {
      console.error('[askGroq] response JSON parse error', { callId, textPreview: text.slice(0, 300), error: parseErr instanceof Error ? { name: parseErr.name, message: parseErr.message } : parseErr })
      throw new Error('Invalid JSON from Groq API')
    }

    if (!response.ok) {
      console.error('[askGroq] http error', { callId, status: response.status, statusText: response.statusText, data })
      throw new Error(data.error?.message || `Error from Groq API (${response.status})`);
    }

    const content = data?.choices?.[0]?.message?.content?.trim?.()
    const endedAt = Date.now()
    console.log('[askGroq] success', { callId, durationMs: endedAt - startedAt, contentPreview: String(content).slice(0, 200) })
    return content
  } catch (error) {
    const endedAt = Date.now()
    console.error('[askGroq] error', { callId, durationMs: endedAt - startedAt, error: error instanceof Error ? { name: error.name, message: error.message, stack: error.stack } : error })
    return 'Sorry, something went wrong while generating the response.';
  }
}
