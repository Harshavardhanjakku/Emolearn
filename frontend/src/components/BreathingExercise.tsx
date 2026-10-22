import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  Brain,
  Check,
  Clock3,
  Edit3,
  Lightbulb,
  Pause,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Trash2,
  Volume2,
  VolumeX,
  Zap,
} from "lucide-react";
import quotesData from "./quotes.json";
import loFiMusic from "../assets/lofi.mp3";

interface BreathingExerciseProps {
  isOpen: boolean;
  onClose: () => void;
}

type ReflexState = "ready" | "waiting" | "go" | "too-early" | "complete";

const presets = [
  { label: "Quick reset", seconds: 60 },
  { label: "Mindful break", seconds: 120 },
  { label: "Deep recharge", seconds: 300 },
];

const formatTime = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

const BreathingExercise = ({ isOpen, onClose }: BreathingExerciseProps) => {
  const [quotes, setQuotes] = useState<string[]>(quotesData);
  const [quote, setQuote] = useState(quotesData[0] ?? "");
  const [newQuote, setNewQuote] = useState("");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [timer, setTimer] = useState(120);
  const [initialTimer, setInitialTimer] = useState(120);
  const [isPaused, setIsPaused] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [reflexState, setReflexState] = useState<ReflexState>("ready");
  const [reactionTime, setReactionTime] = useState<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const reflexTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const quotesRef = useRef(quotes);
  quotesRef.current = quotes;

  useEffect(() => {
    if (!isOpen) return;
    const availableQuotes = quotesRef.current;
    const firstQuote = availableQuotes[Math.floor(Math.random() * availableQuotes.length)] ?? "";
    setQuote(firstQuote);
    setTimer(120);
    setInitialTimer(120);
    setIsPaused(false);
    setReflexState("ready");
    setReactionTime(null);
    audioRef.current?.pause();
    setIsPlaying(false);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || isPaused || timer <= 0) return;
    const interval = window.setInterval(() => setTimer((value) => Math.max(value - 1, 0)), 1000);
    return () => window.clearInterval(interval);
  }, [isOpen, isPaused, timer]);

  useEffect(() => () => {
    if (reflexTimeoutRef.current) window.clearTimeout(reflexTimeoutRef.current);
  }, []);

  const selectPreset = (seconds: number) => {
    setTimer(seconds);
    setInitialTimer(seconds);
    setIsPaused(false);
  };

  const toggleMusic = () => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      void audioRef.current.play();
      setIsPlaying(true);
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const startReflexTest = () => {
    if (reflexTimeoutRef.current) window.clearTimeout(reflexTimeoutRef.current);
    setReactionTime(null);
    setReflexState("waiting");
    reflexTimeoutRef.current = window.setTimeout(() => {
      setReflexState("go");
      startTimeRef.current = performance.now();
    }, 1800 + Math.random() * 2200);
  };

  const handleReflexClick = () => {
    if (reflexState === "waiting") {
      if (reflexTimeoutRef.current) window.clearTimeout(reflexTimeoutRef.current);
      setReflexState("too-early");
      return;
    }
    if (reflexState === "go" && startTimeRef.current) {
      setReactionTime(Math.round(performance.now() - startTimeRef.current));
      setReflexState("complete");
    }
  };

  const saveQuote = () => {
    const trimmed = newQuote.trim();
    if (!trimmed) return;
    setQuotes((items) =>
      editingIndex === null
        ? [...items, trimmed]
        : items.map((item, index) => (index === editingIndex ? trimmed : item))
    );
    setQuote(trimmed);
    setNewQuote("");
    setEditingIndex(null);
  };

  const editQuote = (index: number) => {
    setNewQuote(quotes[index]);
    setEditingIndex(index);
  };

  const deleteQuote = (index: number) => {
    setQuotes((items) => items.filter((_, itemIndex) => itemIndex !== index));
    if (index === editingIndex) {
      setEditingIndex(null);
      setNewQuote("");
    }
  };

  if (!isOpen) return null;

  const progress = initialTimer ? ((initialTimer - timer) / initialTimer) * 100 : 0;
  const reflexLabel = {
    ready: "Ready when you are",
    waiting: "Wait for the signal...",
    go: "Tap now!",
    "too-early": "Too early — try again",
    complete: `${reactionTime} ms reaction`,
  }[reflexState];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-background/95 p-4 text-foreground backdrop-blur-md sm:p-6">
      <audio ref={audioRef} src={loFiMusic} loop />
      <div className="mx-auto max-w-5xl">
        <header className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-md">
              <Brain className="h-5 w-5" />
            </div>
            <div>
              <p className="page-kicker">Recharge</p>
              <h2 className="font-display text-2xl font-bold">Brain Reset</h2>
            </div>
          </div>
          <button type="button" onClick={onClose} className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground" aria-label="Close break">
            <ArrowLeft className="h-4 w-4" /> Back
          </button>
        </header>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <main className="space-y-5">
            <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-primary">
                    <Clock3 className="h-4 w-4" /> Your reset timer
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">Take a moment to pause, breathe, and return refreshed.</p>
                </div>
                <button type="button" onClick={toggleMusic} className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-xs font-medium transition-colors hover:bg-muted" aria-pressed={isPlaying}>
                  {isPlaying ? <Volume2 className="h-4 w-4 text-primary" /> : <VolumeX className="h-4 w-4 text-muted-foreground" />}
                  {isPlaying ? "Music on" : "Play music"}
                </button>
              </div>
              <div className="mt-6 flex flex-col items-center">
                <div className="relative flex h-44 w-44 items-center justify-center rounded-full border-[10px] border-muted">
                  <div className="absolute inset-[-10px] rounded-full border-[10px] border-primary border-l-transparent border-b-transparent transition-transform" style={{ transform: `rotate(${progress * 3.6}deg)` }} />
                  <div className="text-center">
                    <p className="font-mono text-4xl font-bold tracking-tight">{formatTime(timer)}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{timer === 0 ? "Reset complete" : isPaused ? "Paused" : "In progress"}</p>
                  </div>
                </div>
                <div className="mt-5 flex items-center gap-2">
                  <button type="button" onClick={() => setIsPaused((value) => !value)} disabled={timer === 0} className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50">
                    {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />} {isPaused ? "Resume" : "Pause"}
                  </button>
                  <button type="button" onClick={() => selectPreset(initialTimer)} className="inline-flex items-center gap-2 rounded-md border border-border px-4 py-2 text-sm font-semibold transition-colors hover:bg-muted">
                    <RotateCcw className="h-4 w-4" /> Restart
                  </button>
                </div>
                <div className="mt-5 flex flex-wrap justify-center gap-2">
                  {presets.map((preset) => (
                    <button key={preset.seconds} type="button" onClick={() => selectPreset(preset.seconds)} className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${initialTimer === preset.seconds ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:bg-muted"}`}>
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-primary"><Sparkles className="h-4 w-4" /> Motivation</div>
                  <p className="mt-1 text-sm text-muted-foreground">A small reminder for your next step.</p>
                </div>
                <button type="button" onClick={() => setQuote(quotes[Math.floor(Math.random() * quotes.length)] ?? "")} className="rounded-md border border-border p-2 text-muted-foreground hover:bg-muted hover:text-foreground" aria-label="New motivation">
                  <RefreshCw className="h-4 w-4" />
                </button>
              </div>
              <blockquote className="mt-4 border-l-2 border-primary pl-4 text-lg font-medium leading-relaxed">“{quote}”</blockquote>
            </section>

            <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-primary"><Zap className="h-4 w-4" /> Reflex reset</div>
                  <p className="mt-1 text-sm text-muted-foreground">A quick reaction challenge to bring your focus back.</p>
                </div>
                <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">{reflexState === "complete" ? "Complete" : "Mini game"}</span>
              </div>
              <motion.button type="button" onClick={reflexState === "ready" || reflexState === "too-early" || reflexState === "complete" ? startReflexTest : handleReflexClick} whileTap={{ scale: 0.98 }} className={`mt-4 flex min-h-24 w-full flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors ${reflexState === "go" ? "border-success bg-success/15 text-success" : "border-border bg-muted/40 hover:bg-muted"}`}>
                <span className="text-base font-semibold">{reflexLabel}</span>
                <span className="mt-1 text-xs text-muted-foreground">{reflexState === "go" ? "Click as soon as the panel changes" : reflexState === "complete" ? "Start again for another attempt" : "Train your attention in seconds"}</span>
              </motion.button>
            </section>
          </main>

          <aside className="rounded-xl border border-border bg-card p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <Edit3 className="h-4 w-4 text-primary" />
              <h3 className="font-display font-bold">Manage quotes</h3>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">Save reminders that help you keep going.</p>
            <div className="mt-4 flex gap-2">
              <input value={newQuote} onChange={(event) => setNewQuote(event.target.value)} onKeyDown={(event) => event.key === "Enter" && saveQuote()} placeholder="Add a quote..." className="min-w-0 flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-ring" />
              <button type="button" onClick={saveQuote} className="rounded-md bg-primary px-3 text-primary-foreground hover:bg-primary/90" aria-label={editingIndex === null ? "Add quote" : "Save quote"}>
                {editingIndex === null ? <Plus className="h-4 w-4" /> : <Check className="h-4 w-4" />}
              </button>
            </div>
            {editingIndex !== null && <button type="button" onClick={() => { setEditingIndex(null); setNewQuote(""); }} className="mt-2 text-xs text-muted-foreground hover:text-foreground">Cancel editing</button>}
            <div className="mt-4 max-h-80 space-y-2 overflow-y-auto pr-1">
              {quotes.map((item, index) => (
                <div key={`${item}-${index}`} className="group rounded-md border border-border bg-muted/40 p-3">
                  <p className="text-sm leading-relaxed">{item}</p>
                  <div className="mt-2 flex gap-1 opacity-70 transition-opacity group-hover:opacity-100">
                    <button type="button" onClick={() => { setQuote(item); }} className="rounded p-1.5 text-muted-foreground hover:bg-background hover:text-foreground" aria-label="Use quote"><Lightbulb className="h-3.5 w-3.5" /></button>
                    <button type="button" onClick={() => editQuote(index)} className="rounded p-1.5 text-muted-foreground hover:bg-background hover:text-foreground" aria-label="Edit quote"><Edit3 className="h-3.5 w-3.5" /></button>
                    <button type="button" onClick={() => deleteQuote(index)} className="rounded p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive" aria-label="Delete quote"><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                </div>
              ))}
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
};

export default BreathingExercise;
