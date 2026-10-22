import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CalendarDays,
  Clock3,
  Donut,
  Gauge,
  Minus,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmotionTracker } from "@/components/EmotionTracker";
import { API_BASE, emotionToColor, emotionToEmoji } from "@/lib/emotions";

type RangeDays = 7 | 30 | 90 | 365;
type HistoryRecord = {
  time: string;
  emotion: string;
  value: number | null;
  source: string[];
  facial_emotion?: string | null;
  voice_emotion?: string | null;
  interaction_emotion?: string | null;
};

const RANGE_OPTIONS: Array<{ label: string; value: RangeDays }> = [
  { label: "7 days", value: 7 },
  { label: "30 days", value: 30 },
  { label: "90 days", value: 90 },
  { label: "1 year", value: 365 },
];

const tooltipStyle = {
  backgroundColor: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: "10px",
  color: "hsl(var(--foreground))",
};

const formatDate = (date: Date) =>
  date.toLocaleDateString([], { month: "short", day: "numeric" });

const formatTime = (date: Date) =>
  date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const startOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate());

const percentage = (value: number, total: number) => (total ? Math.round((value / total) * 100) : 0);

const ChangeIndicator = ({ current, previous }: { current: number; previous: number }) => {
  if (!previous) return <span className="text-xs text-muted-foreground">No prior data</span>;
  const change = Math.round(((current - previous) / previous) * 100);
  const Icon = change > 0 ? TrendingUp : change < 0 ? TrendingDown : Minus;
  return (
    <span className={`inline-flex items-center gap-1 text-xs ${change > 0 ? "text-success" : change < 0 ? "text-destructive" : "text-muted-foreground"}`}>
      <Icon className="h-3 w-3" /> {Math.abs(change)}% vs prior period
    </span>
  );
};

const Insights = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [range, setRange] = useState<RangeDays>(30);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const loadHistory = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/emotion/history`);
        if (!response.ok) throw new Error(`History request failed: ${response.status}`);
        const data = await response.json();
        if (!cancelled) setRecords(Array.isArray(data) ? data : []);
      } catch (requestError) {
        console.error("Failed to load emotion history:", requestError);
        if (!cancelled) setError("Emotion history is unavailable. Start the backend and try again.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void loadHistory();
    return () => {
      cancelled = true;
    };
  }, []);

  const parsedRecords = useMemo(
    () =>
      records
        .map((record) => ({ ...record, date: new Date(record.time) }))
        .filter((record) => !Number.isNaN(record.date.getTime()))
        .sort((a, b) => a.date.getTime() - b.date.getTime()),
    [records]
  );

  const latestDate = useMemo(() => parsedRecords.at(-1)?.date ?? new Date(), [parsedRecords]);
  const currentStart = new Date(latestDate.getTime() - range * 86400000);
  const previousStart = new Date(currentStart.getTime() - range * 86400000);
  const currentRecords = parsedRecords.filter(({ date }) => date >= currentStart && date <= latestDate);
  const previousRecords = parsedRecords.filter(({ date }) => date >= previousStart && date < currentStart);

  const emotionCounts = useMemo(() => {
    const counts = new Map<string, number>();
    currentRecords.forEach(({ emotion }) => counts.set(emotion || "Unknown", (counts.get(emotion || "Unknown") ?? 0) + 1));
    return [...counts.entries()]
      .map(([emotion, count]) => ({ emotion, count, fill: emotionToColor(emotion), emoji: emotionToEmoji(emotion) }))
      .sort((a, b) => b.count - a.count);
  }, [currentRecords]);

  const previousCounts = useMemo(() => {
    const counts = new Map<string, number>();
    previousRecords.forEach(({ emotion }) => counts.set(emotion || "Unknown", (counts.get(emotion || "Unknown") ?? 0) + 1));
    return counts;
  }, [previousRecords]);

  const dominantEmotion = emotionCounts[0]?.emotion ?? "—";
  const engagement = percentage(currentRecords.filter(({ emotion }) => emotion === "Engaged").length, currentRecords.length);
  const intensityRecords = currentRecords.filter(({ value }) => typeof value === "number");
  const averageIntensity = intensityRecords.length
    ? Math.round((intensityRecords.reduce((sum, record) => sum + (record.value ?? 0), 0) / intensityRecords.length) * 100)
    : null;

  const trendData = useMemo(() => {
    const bucket = new Map<string, { label: string; timestamp: number; total: number; intensity: number; count: number; engaged: number }>();
    currentRecords.forEach((record) => {
      const day = startOfDay(record.date);
      const key = day.toISOString();
      const item = bucket.get(key) ?? { label: formatDate(day), timestamp: day.getTime(), total: 0, intensity: 0, count: 0, engaged: 0 };
      item.total += 1;
      if (typeof record.value === "number") {
        item.intensity += record.value;
        item.count += 1;
      }
      if (record.emotion === "Engaged") item.engaged += 1;
      bucket.set(key, item);
    });
    return [...bucket.values()].sort((a, b) => a.timestamp - b.timestamp).map((item) => ({
      ...item,
      intensity: item.count ? Math.round((item.intensity / item.count) * 100) : null,
      engagement: percentage(item.engaged, item.total),
    }));
  }, [currentRecords]);

  const hourlyData = useMemo(() => {
    const buckets = Array.from({ length: 24 }, (_, hour) => ({ hour, label: `${String(hour).padStart(2, "0")}:00`, total: 0, engaged: 0 }));
    currentRecords.forEach(({ date, emotion }) => {
      buckets[date.getHours()].total += 1;
      if (emotion === "Engaged") buckets[date.getHours()].engaged += 1;
    });
    return buckets.filter((bucket) => bucket.total > 0).map((bucket) => ({ ...bucket, engagement: percentage(bucket.engaged, bucket.total) }));
  }, [currentRecords]);

  const comparisonData = emotionCounts.map(({ emotion, count, fill }) => ({
    emotion,
    current: count,
    previous: previousCounts.get(emotion) ?? 0,
    fill,
  }));

  const monthlyData = useMemo(() => {
    const months = new Map<string, { label: string; timestamp: number; total: number; engaged: number }>();
    parsedRecords.filter(({ date }) => date >= new Date(latestDate.getTime() - 365 * 86400000)).forEach(({ date, emotion }) => {
      const month = new Date(date.getFullYear(), date.getMonth(), 1);
      const key = month.toISOString();
      const item = months.get(key) ?? { label: month.toLocaleDateString([], { month: "short", year: "2-digit" }), timestamp: month.getTime(), total: 0, engaged: 0 };
      item.total += 1;
      if (emotion === "Engaged") item.engaged += 1;
      months.set(key, item);
    });
    return [...months.values()].sort((a, b) => a.timestamp - b.timestamp).map((item) => ({ ...item, engagement: percentage(item.engaged, item.total) }));
  }, [latestDate, parsedRecords]);

  const recentActivity = [...currentRecords].reverse().slice(0, 8);

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="page-kicker">Analytics</p>
          <h2 className="page-title">Emotion Insights</h2>
          <p className="mt-2 max-w-2xl text-muted-foreground">A clear view of your learning energy, engagement, and emotional patterns.</p>
        </div>
        <div className="flex flex-wrap gap-1 rounded-lg border border-border bg-muted/40 p-1">
          {RANGE_OPTIONS.map((option) => (
            <Button key={option.value} size="sm" variant={range === option.value ? "default" : "ghost"} onClick={() => setRange(option.value)} className="h-8 px-3 text-xs">
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      <EmotionTracker />

      {loading && <Card className="p-12 text-center text-sm text-muted-foreground">Loading your emotion history…</Card>}
      {!loading && error && <Card className="border-destructive/30 bg-destructive/10 p-10 text-center text-sm text-destructive">{error}</Card>}
      {!loading && !error && currentRecords.length === 0 && (
        <Card className="border-dashed p-12 text-center">
          <Activity className="mx-auto h-8 w-8 text-muted-foreground" />
          <h3 className="mt-3 font-display font-bold">No observations in this period</h3>
          <p className="mt-1 text-sm text-muted-foreground">Try a wider date range or start a camera/microphone session to collect real data.</p>
        </Card>
      )}

      {!loading && !error && currentRecords.length > 0 && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { label: "Total observations", value: currentRecords.length.toLocaleString(), icon: Activity, note: <ChangeIndicator current={currentRecords.length} previous={previousRecords.length} /> },
              { label: "Dominant emotion", value: `${emotionToEmoji(dominantEmotion)} ${dominantEmotion}`, icon: Donut, note: `${percentage(emotionCounts[0]?.count ?? 0, currentRecords.length)}% of observations` },
              { label: "Engagement", value: `${engagement}%`, icon: Gauge, note: "Engaged observations only" },
              { label: "Avg. intensity", value: averageIntensity == null ? "—" : `${averageIntensity}%`, icon: BarChart3, note: "Detection confidence records" },
            ].map((metric) => {
              const Icon = metric.icon;
              return (
                <Card key={metric.label} className="border-border/70 p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-muted-foreground">{metric.label}</p>
                    <Icon className="h-4 w-4 text-primary" />
                  </div>
                  <p className="mt-3 truncate font-display text-2xl font-bold">{metric.value}</p>
                  <div className="mt-1 text-xs text-muted-foreground">{metric.note}</div>
                </Card>
              );
            })}
          </div>

          <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="page-kicker">Distribution</p>
                  <h3 className="font-display text-lg font-bold">Emotion mix</h3>
                </div>
                <Donut className="h-5 w-5 text-primary" />
              </div>
              <div className="mt-4 grid items-center gap-4 sm:grid-cols-[minmax(0,1fr)_10rem]">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={emotionCounts} dataKey="count" nameKey="emotion" innerRadius="58%" outerRadius="82%" paddingAngle={3} stroke="hsl(var(--card))" strokeWidth={3}>
                        {emotionCounts.map((entry) => <Cell key={entry.emotion} fill={entry.fill} />)}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} formatter={(value: number, _name, item) => [`${value} observations`, `${item.payload.emoji} ${item.payload.emotion}`]} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2">
                  {emotionCounts.map((entry) => (
                    <div key={entry.emotion} className="flex items-center justify-between gap-3 text-xs">
                      <span className="flex min-w-0 items-center gap-2"><span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: entry.fill }} /> <span className="truncate">{entry.emoji} {entry.emotion}</span></span>
                      <span className="font-mono text-muted-foreground">{percentage(entry.count, currentRecords.length)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="page-kicker">Intensity</p>
                  <h3 className="font-display text-lg font-bold">Daily emotion trend</h3>
                </div>
                <TrendingUp className="h-5 w-5 text-primary" />
              </div>
              <div className="mt-5 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData}>
                    <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={34} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(value: number | null, name: string) => [value == null ? "No value" : `${value}%`, name === "intensity" ? "Intensity" : "Engagement"]} />
                    <Line type="monotone" connectNulls={false} dataKey="intensity" stroke="hsl(var(--primary))" strokeWidth={3} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="engagement" stroke="hsl(var(--success))" strokeWidth={2} dot={false} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div><p className="page-kicker">Patterns</p><h3 className="font-display text-lg font-bold">Hourly engagement</h3></div>
                <Clock3 className="h-5 w-5 text-primary" />
              </div>
              <div className="mt-5 h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hourlyData}>
                    <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={34} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(value: number) => [`${value}%`, "Engagement"]} />
                    <Bar dataKey="engagement" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div><p className="page-kicker">Comparison</p><h3 className="font-display text-lg font-bold">Current vs previous period</h3></div>
                <CalendarDays className="h-5 w-5 text-primary" />
              </div>
              <div className="mt-5 h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData}>
                    <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                    <XAxis dataKey="emotion" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                    <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={28} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="current" name="Current" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="previous" name="Previous" fill="hsl(var(--muted-foreground))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {monthlyData.length > 0 && (
            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div><p className="page-kicker">Historical view</p><h3 className="font-display text-lg font-bold">Yearly overview</h3><p className="mt-1 text-xs text-muted-foreground">Only months containing real observations are shown.</p></div>
                <CalendarDays className="h-5 w-5 text-primary" />
              </div>
              <div className="mt-5 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={monthlyData}>
                    <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={34} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(value: number, name: string) => [`${value}%`, name === "engagement" ? "Engagement" : "Observations"]} />
                    <Bar dataKey="engagement" name="Engagement" fill="hsl(var(--accent))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          <div className="grid gap-5 lg:grid-cols-[1fr_0.85fr]">
            <Card className="p-5">
              <div className="flex items-center justify-between"><div><p className="page-kicker">Breakdown</p><h3 className="font-display text-lg font-bold">Emotion details</h3></div><Badge variant="outline">{currentRecords.length} records</Badge></div>
              <div className="mt-4 space-y-3">
                {emotionCounts.map((entry) => (
                  <div key={entry.emotion}>
                    <div className="mb-1 flex items-center justify-between text-sm"><span>{entry.emoji} {entry.emotion}</span><span className="font-mono text-xs text-muted-foreground">{entry.count} · {percentage(entry.count, currentRecords.length)}%</span></div>
                    <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full" style={{ width: `${percentage(entry.count, currentRecords.length)}%`, backgroundColor: entry.fill }} /></div>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="p-5">
              <div><p className="page-kicker">Live history</p><h3 className="font-display text-lg font-bold">Recent activity</h3></div>
              <div className="mt-4 space-y-2">
                {recentActivity.map((entry, index) => (
                  <div key={`${entry.time}-${entry.emotion}-${index}`} className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
                    <span className="flex min-w-0 items-center gap-2"><span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: emotionToColor(entry.emotion) }} /><span className="truncate">{emotionToEmoji(entry.emotion)} {entry.emotion}</span></span>
                    <span className="shrink-0 text-xs text-muted-foreground">{formatDate(entry.date)} · {formatTime(entry.date)}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
};

export default Insights;
