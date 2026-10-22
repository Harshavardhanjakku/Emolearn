import { useEffect, useState } from "react";
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "@/components/ui/card";
import { API_BASE, emotionToColor, emotionToEmoji } from "@/lib/emotions";

interface EmotionData {
  time: string;
  facial_emotion: string;
  voice_emotion: string;
  interaction_emotion: string;
  final_emotion: string;
  value: number;
}

const tooltipStyle = {
  backgroundColor: "hsl(var(--card))",
  border: "2px solid hsl(var(--border))",
  borderRadius: "8px",
  color: "hsl(var(--foreground))",
};

export const MoodGraph = () => {
  const [emotionData, setEmotionData] = useState<EmotionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchEmotionData = async () => {
      try {
        const allDataResponse = await fetch(`${API_BASE}/api/emotion/all`);
        if (allDataResponse.ok) {
          const allData = await allDataResponse.json();
          if (!cancelled) {
            setEmotionData(Array.isArray(allData) ? allData.slice(-20) : []);
            setFetchError(false);
          }
        } else {
          const latest = await fetch(`${API_BASE}/api/emotion/latest`);
          if (latest.ok) {
            const latestData = await latest.json();
            if (!cancelled) {
              setEmotionData(latestData?.time ? [latestData] : []);
              setFetchError(false);
            }
          } else if (!cancelled) {
            setFetchError(true);
          }
        }
      } catch {
        if (!cancelled) setFetchError(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchEmotionData();
    const interval = setInterval(fetchEmotionData, 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const chartData = emotionData.map((entry) => ({
    time: entry.time ? new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—",
    value: typeof entry.value === "number" ? entry.value : 0,
    emotion: entry.final_emotion,
    emoji: emotionToEmoji(entry.final_emotion),
    facial: entry.facial_emotion,
    voice: entry.voice_emotion,
  }));

  const emotionDistribution = emotionData.reduce(
    (acc, entry) => {
      const emotion = entry.final_emotion || "Unknown";
      acc[emotion] = (acc[emotion] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  const distributionData = Object.entries(emotionDistribution).map(([emotion, count]) => ({
    emotion,
    count,
    emoji: emotionToEmoji(emotion),
    fill: emotionToColor(emotion),
  }));

  return (
    <Card className="p-5">
      <div className="mb-4">
        <p className="page-kicker">History</p>
        <h3 className="font-display text-lg font-bold">Mood trends</h3>
        <p className="text-xs text-muted-foreground">Recorded readings from the emotion API — not sample data.</p>
      </div>

      {loading && <p className="py-10 text-center text-sm text-muted-foreground">Loading emotion history…</p>}

      {!loading && (fetchError || emotionData.length === 0) && (
        <div className="rounded-md border-2 border-dashed border-border p-8 text-center text-sm text-muted-foreground">
          {fetchError
            ? "Emotion history is unavailable because the backend did not respond."
            : "Not enough recorded readings yet. Start a session with camera or microphone to collect history."}
        </div>
      )}

      {!loading && emotionData.length > 0 && (
        <div className="space-y-6">
          <div>
            <h4 className="mb-2 text-sm font-medium">Intensity over recent readings</h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }} width={28} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(value: number, _name, props: { payload?: { emoji: string; emotion: string } }) => [
                      `${props.payload?.emoji ?? ""} ${props.payload?.emotion ?? ""} (${Math.round(value * 100)}%)`,
                      "Intensity",
                    ]}
                  />
                  <Line type="monotone" dataKey="value" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {distributionData.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-medium">Distribution</h4>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={distributionData}>
                    <XAxis dataKey="emoji" axisLine={false} tickLine={false} />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value: number, _name, props: { payload?: { emotion: string } }) => [
                        `${value} readings`,
                        props.payload?.emotion,
                      ]}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div>
            <h4 className="mb-2 text-sm font-medium">Recent events</h4>
            <ul className="max-h-40 space-y-2 overflow-y-auto">
              {chartData
                .slice()
                .reverse()
                .map((entry, index) => (
                  <li
                    key={`${entry.time}-${index}`}
                    className="flex items-center justify-between rounded-md border border-border bg-muted/40 px-3 py-2 text-sm"
                  >
                    <span>
                      <span aria-hidden>{entry.emoji}</span> {entry.emotion}
                    </span>
                    <span className="text-xs text-muted-foreground">{entry.time}</span>
                  </li>
                ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
};
