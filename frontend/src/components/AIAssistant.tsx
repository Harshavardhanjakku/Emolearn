import { useEffect, useRef, useState } from "react";
import {
  Bot,
  Brain,
  ChevronDown,
  Gamepad2,
  Lightbulb,
  Loader2,
  Maximize2,
  MessageCircle,
  Minimize2,
  Send,
  Sparkles,
  X,
  Youtube,
} from "lucide-react";
import { askGroq } from "@/api/groqGPT";
import { useSession } from "@/contexts/session-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AITutorSection } from "./AITutorSection";
import { GamesSection } from "./GamesSection";
import { YouTubeSummarizer } from "./YouTubeSummarizer";
import VoiceInput from "./VoiceInput";
import "./voice.css";

interface AIAssistantProps {
  isOpen: boolean;
  onClose: () => void;
  currentEmotion: string;
  isEmotionTriggered?: boolean;
}

type ChatMessage = {
  type: "user" | "ai";
  text: string;
  timestamp: string;
};

type AssistantView = "chat" | "youtube" | "games" | "tutor";

const suggestions = [
  { label: "Explain a concept", prompt: "Explain this concept in simple terms with an example.", icon: Lightbulb },
  { label: "Ask a coding question", prompt: "Help me understand this coding question: ", icon: Brain },
  { label: "Generate a quiz", prompt: "Create a short quiz to test my understanding of ", icon: Sparkles },
  { label: "Summarize a lesson", prompt: "Summarize this lesson: ", icon: Youtube },
];

const timeLabel = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const formatMessageText = (text: string) => {
  const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const formatted = escaped
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br />");
  return <span dangerouslySetInnerHTML={{ __html: formatted }} />;
};

export const AIAssistant = ({ isOpen, onClose, currentEmotion, isEmotionTriggered }: AIAssistantProps) => {
  const { handleManualAssistantToggle } = useSession();
  const [message, setMessage] = useState("");
  const [isMinimized, setIsMinimized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState<AssistantView>("chat");
  const [conversation, setConversation] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setIsMinimized(false);
      setView("chat");
      setError("");
      inputRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && conversation.length === 0) {
      setConversation([
        {
          type: "ai",
          text: isEmotionTriggered
            ? `I noticed your current mood is ${currentEmotion}. I’m here to help you learn at your pace. What would you like to work through?`
            : "Hi! I’m your AI learning companion. Ask me to explain a concept, help with code, make a quiz, or summarize a lesson.",
          timestamp: timeLabel(),
        },
      ]);
    }
  }, [conversation.length, currentEmotion, isEmotionTriggered, isOpen]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation, isLoading]);

  const addAssistantMessage = (text: string) => {
    setConversation((previous) => [...previous, { type: "ai", text, timestamp: timeLabel() }]);
  };

  const sendPrompt = async (prompt: string, mode = "simple") => {
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setError("");
    setConversation((previous) => [...previous, { type: "user", text: trimmed, timestamp: timeLabel() }]);
    setMessage("");
    setIsLoading(true);

    try {
      const detectedMode =
        trimmed.includes("youtube.com") || trimmed.includes("youtu.be") ? "youtube" : mode;
      const response = await askGroq(trimmed, detectedMode);
      if (!response?.trim()) throw new Error("The assistant returned an empty response.");
      addAssistantMessage(response);
    } catch (requestError) {
      console.error("[AIAssistant] request failed", requestError);
      setError("I couldn’t complete that request. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestion = (prompt: string) => {
    setMessage(prompt);
    inputRef.current?.focus();
  };

  const handleVoiceResult = (text: string) => {
    setMessage(text);
    void sendPrompt(text);
  };

  const renderToolView = () => {
    if (view === "youtube") {
      return (
        <YouTubeSummarizer
          onSummaryComplete={(summary) => {
            setView("chat");
            addAssistantMessage(summary);
          }}
          onClose={() => setView("chat")}
        />
      );
    }
    if (view === "games") return <GamesSection onClose={() => setView("chat")} />;
    if (view === "tutor") return <AITutorSection onClose={() => setView("chat")} />;
    return null;
  };

  return (
    <>
      <button
        type="button"
        onClick={handleManualAssistantToggle}
        aria-label={isOpen ? "Close AI learning assistant" : "Open AI learning assistant"}
        aria-expanded={isOpen}
        className="fixed bottom-5 right-5 z-40 flex h-14 w-14 items-center justify-center rounded-full border-2 border-primary/30 bg-primary text-primary-foreground shadow-[0_10px_30px_hsl(var(--primary)/0.35)] transition-all duration-300 hover:-translate-y-1 hover:scale-105 hover:shadow-[0_14px_36px_hsl(var(--primary)/0.45)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:bottom-6 sm:right-6"
      >
        {isOpen ? <ChevronDown className="h-6 w-6" /> : <MessageCircle className="h-6 w-6" />}
        {!isOpen && isEmotionTriggered && (
          <span className="absolute right-0 top-0 h-3.5 w-3.5 rounded-full border-2 border-background bg-destructive" aria-label="New AI suggestion" />
        )}
      </button>

      <section
        aria-label="AI learning assistant"
        aria-hidden={!isOpen}
        className={`fixed bottom-24 right-4 z-40 w-[calc(100vw-2rem)] max-w-[27rem] origin-bottom-right overflow-hidden rounded-2xl border border-border bg-card text-card-foreground shadow-2xl transition-all duration-300 sm:right-6 ${
          isOpen ? "translate-y-0 scale-100 opacity-100" : "pointer-events-none translate-y-3 scale-95 opacity-0"
        }`}
      >
        <header className="flex items-center justify-between border-b border-border bg-muted/40 px-4 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent text-primary-foreground shadow-md">
              <Bot className="h-5 w-5" />
              <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-card bg-success" />
            </div>
            <div className="min-w-0">
              <h2 className="truncate font-display text-sm font-bold">EmoLearn AI</h2>
              <p className="flex items-center gap-1 text-xs text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full bg-success" /> Online · Learning companion
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setIsMinimized(true)} aria-label="Minimize AI assistant">
              <Minimize2 className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose} aria-label="Close AI assistant">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </header>

        {isMinimized ? (
          <button type="button" className="flex w-full items-center gap-2 p-4 text-left text-sm hover:bg-muted" onClick={() => setIsMinimized(false)}>
            <Maximize2 className="h-4 w-4" /> Restore conversation
          </button>
        ) : (
          <>
            {view !== "chat" ? (
              <div className="max-h-[32rem] overflow-y-auto p-4">{renderToolView()}</div>
            ) : (
              <>
                <div className="flex max-h-80 min-h-56 flex-col gap-3 overflow-y-auto p-4" role="log" aria-live="polite">
                  {conversation.length === 0 && (
                    <div className="m-auto max-w-xs text-center">
                      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                        <Sparkles className="h-6 w-6" />
                      </div>
                      <h3 className="font-display font-bold">What are you learning today?</h3>
                      <p className="mt-1 text-xs text-muted-foreground">I can explain concepts, help with code, create quizzes, and summarize lessons.</p>
                    </div>
                  )}
                  {conversation.map((item, index) => (
                    <div key={`${item.timestamp}-${index}`} className={`flex ${item.type === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm ${item.type === "user" ? "rounded-br-sm bg-primary text-primary-foreground" : "rounded-bl-sm bg-muted text-foreground"}`}>
                        {item.type === "ai" && <Bot className="mb-1 inline-block h-3.5 w-3.5 text-primary" aria-hidden />}
                        <span className="ml-1">{formatMessageText(item.text)}</span>
                        <p className="mt-1 text-right text-[10px] opacity-60">{item.timestamp}</p>
                      </div>
                    </div>
                  ))}
                  {isLoading && (
                    <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-muted px-4 py-3 text-muted-foreground" aria-label="AI is typing">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.2s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.1s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {error && <p className="px-4 pb-2 text-xs text-destructive" role="alert">{error}</p>}

                <div className="flex gap-2 overflow-x-auto border-t border-border px-4 py-3">
                  {suggestions.map((suggestion) => {
                    const Icon = suggestion.icon;
                    return (
                      <button key={suggestion.label} type="button" onClick={() => handleSuggestion(suggestion.prompt)} className="flex shrink-0 items-center gap-1.5 rounded-full border border-border bg-background px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground">
                        <Icon className="h-3 w-3 text-primary" /> {suggestion.label}
                      </button>
                    );
                  })}
                </div>

                <div className="flex items-center gap-2 border-t border-border bg-muted/20 p-3">
                  <Input
                    ref={inputRef}
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void sendPrompt(message);
                      }
                      if (event.key === "Escape") onClose();
                    }}
                    placeholder="Ask a learning question..."
                    aria-label="Message for AI assistant"
                    disabled={isLoading}
                    className="h-10 rounded-xl bg-background"
                  />
                  <VoiceInput onResult={handleVoiceResult} />
                  <Button type="button" size="icon" className="h-10 w-10 shrink-0 rounded-xl" onClick={() => void sendPrompt(message)} disabled={isLoading || !message.trim()} aria-label="Send message">
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  </Button>
                </div>
                <div className="flex items-center justify-center gap-3 border-t border-border px-3 py-2">
                  <button type="button" onClick={() => setView("tutor")} className="text-[11px] text-muted-foreground hover:text-foreground">AI Tutor</button>
                  <button type="button" onClick={() => setView("games")} className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"><Gamepad2 className="h-3 w-3" /> Games</button>
                  <button type="button" onClick={() => setView("youtube")} className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"><Youtube className="h-3 w-3" /> Summarize video</button>
                </div>
              </>
            )}
          </>
        )}
      </section>
    </>
  );
};
