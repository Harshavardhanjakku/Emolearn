import { Card } from "@/components/ui/card";
import { useSession } from "@/contexts/session-context";
import { EMOTION_LABELS, emotionToEmoji } from "@/lib/emotions";
import { cn } from "@/lib/utils";

export const EmotionTracker = () => {
  const { currentEmotion, detectionStatus } = useSession();
  const isActive = detectionStatus === "active";

  return (
    <Card className="p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="page-kicker">Now</p>
          <h3 className="font-display text-lg font-bold">Current emotion</h3>
        </div>
        <span
          className={cn(
            "rounded-md border-2 px-2 py-1 text-xs font-semibold",
            isActive ? "border-success/40 bg-success/15 text-success" : "border-border text-muted-foreground"
          )}
        >
          {detectionStatus === "active"
            ? "Detecting"
            : detectionStatus === "initializing"
              ? "Starting"
              : detectionStatus === "error"
                ? "Unavailable"
                : "Off"}
        </span>
      </div>

      {currentEmotion && isActive ? (
        <div className="mb-4 rounded-md border-2 border-foreground/15 bg-muted/50 p-4 text-center">
          <div className="text-3xl" aria-hidden>
            {emotionToEmoji(currentEmotion)}
          </div>
          <p className="mt-1 font-display text-lg font-semibold">{currentEmotion}</p>
          <p className="text-xs text-muted-foreground">From the live detection pipeline</p>
        </div>
      ) : (
        <div className="mb-4 rounded-md border-2 border-dashed border-border p-4 text-center text-sm text-muted-foreground">
          {detectionStatus === "error"
            ? "No live reading. The emotion service did not respond."
            : "Enable camera or microphone on Learning or Focus to start detection. Nothing is shown as live until a reading arrives."}
        </div>
      )}

      <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
        {EMOTION_LABELS.map((label) => {
          const selected = currentEmotion === label && isActive;
          return (
            <div
              key={label}
              className={cn(
                "rounded-md border-2 px-1 py-2 text-center",
                selected ? "border-primary bg-primary/10" : "border-border bg-card"
              )}
            >
              <div className="text-lg" aria-hidden>
                {emotionToEmoji(label)}
              </div>
              <p className="mt-1 truncate text-[10px] font-medium">{label === "Face Not Detected" ? "No face" : label}</p>
            </div>
          );
        })}
      </div>
    </Card>
  );
};
