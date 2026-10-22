import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { askGroq } from "@/api/groqGPT";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useSession } from "@/contexts/session-context";

export const QuizDialog = () => {
  const { isQuizOpen, setIsQuizOpen, currentLesson } = useSession();
  const [content, setContent] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => {
    if (!isQuizOpen) {
      setStatus("idle");
      setContent("");
      return;
    }

    let cancelled = false;
    const run = async () => {
      setStatus("loading");
      try {
        const result = await askGroq(currentLesson.chapter, "quiz");
        if (cancelled) return;
        setContent(String(result || "No quiz was returned."));
        setStatus("ready");
      } catch (error) {
        console.error("Quiz generation failed:", error);
        if (cancelled) return;
        setContent("Quiz generation is unavailable right now. Check the AI configuration and try again.");
        setStatus("error");
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [currentLesson.chapter, isQuizOpen]);

  return (
    <Dialog open={isQuizOpen} onOpenChange={setIsQuizOpen}>
      <DialogContent className="max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Quiz · {currentLesson.title}</DialogTitle>
          <DialogDescription>
            Generated from the current lesson. Answers are shown with the questions when the tutor returns them.
          </DialogDescription>
        </DialogHeader>
        {status === "loading" && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Building quiz…
          </div>
        )}
        {(status === "ready" || status === "error") && (
          <pre className="whitespace-pre-wrap rounded-md border-2 border-border bg-muted/40 p-3 font-sans text-sm">
            {content}
          </pre>
        )}
        <div className="flex justify-end">
          <Button type="button" variant="outline" onClick={() => setIsQuizOpen(false)}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
