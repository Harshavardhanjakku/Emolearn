export const EMOTION_LABELS = [
  "Happy",
  "Confused",
  "Frustrated",
  "Sleepy",
  "Bored",
  "Engaged",
  "Face Not Detected",
  "Unknown",
] as const;

export type EmotionLabel = (typeof EMOTION_LABELS)[number] | string;

export const emotionToEmoji = (emotion: string | null): string => {
  switch (emotion) {
    case "Confused":
      return "🤔";
    case "Frustrated":
      return "😫";
    case "Sleepy":
      return "💤";
    case "Bored":
      return "😐";
    case "Engaged":
      return "🎯";
    case "Happy":
      return "😊";
    case "Unknown":
      return "❓";
    case "Face Not Detected":
      return "👤";
    default:
      return emotion ? "❓" : "—";
  }
};

export const emotionToColor = (emotion: string): string => {
  switch (emotion) {
    case "Confused":
      return "#d97706";
    case "Frustrated":
      return "#dc2626";
    case "Sleepy":
      return "#7c3aed";
    case "Bored":
      return "#64748b";
    case "Engaged":
      return "#0f766e";
    case "Happy":
      return "#4f46e5";
    case "Unknown":
      return "#94a3b8";
    case "Face Not Detected":
      return "#ea580c";
    default:
      return "#94a3b8";
  }
};

export const NEGATIVE_EMOTIONS = ["Confused", "Frustrated", "Sleepy", "Bored"];

export const API_BASE = "http://localhost:8000";
