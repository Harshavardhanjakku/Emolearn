import { ChevronLeft, ChevronRight, Maximize, Unlock } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useSession } from "@/contexts/session-context";
import { lessons } from "@/lib/lessons";
import { cn } from "@/lib/utils";

export const VideoPlayer = () => {
  const { lessonIndex, setLessonIndex, isFocusMode, setIsFocusMode, nextLesson, prevLesson, currentLesson } =
    useSession();
  const videoRef = useRef<HTMLDivElement>(null);
  const [unavailable, setUnavailable] = useState(false);

  const requestFullScreen = (element: HTMLElement) => {
    if (element.requestFullscreen) element.requestFullscreen();
    else if ((element as HTMLElement & { webkitRequestFullscreen?: () => void }).webkitRequestFullscreen) {
      (element as HTMLElement & { webkitRequestFullscreen: () => void }).webkitRequestFullscreen();
    }
  };

  const exitFullScreen = () => {
    if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen();
    else if ((document as Document & { webkitExitFullscreen?: () => void }).webkitExitFullscreen) {
      (document as Document & { webkitExitFullscreen: () => void }).webkitExitFullscreen();
    }
  };

  const enterFocus = () => {
    setIsFocusMode(true);
    if (videoRef.current) requestFullScreen(videoRef.current);
  };

  const exitFocus = () => {
    setIsFocusMode(false);
    exitFullScreen();
  };

  useEffect(() => {
    const handleFullScreenChange = () => {
      const isFS =
        document.fullscreenElement === videoRef.current ||
        (document as Document & { webkitFullscreenElement?: Element }).webkitFullscreenElement ===
          videoRef.current;
      setIsFocusMode(!!isFS);
    };

    document.addEventListener("fullscreenchange", handleFullScreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullScreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullScreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullScreenChange);
    };
  }, [setIsFocusMode]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isFocusMode) exitFocus();
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === "ArrowRight") nextLesson();
      if (event.key === "ArrowLeft") prevLesson();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isFocusMode, nextLesson, prevLesson]);

  useEffect(() => {
    setUnavailable(false);
  }, [currentLesson.id]);

  if (!currentLesson?.id) {
    return (
      <div className="surface rounded-lg p-8 text-center">
        <p className="font-display text-lg font-semibold">Lesson unavailable</p>
        <p className="mt-1 text-sm text-muted-foreground">This course has no playable lesson selected.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div
        ref={videoRef}
        className={cn(
          "relative overflow-hidden rounded-lg border-2 border-foreground/20 bg-zinc-950",
          isFocusMode && "fixed inset-0 z-50 rounded-none border-0"
        )}
      >
        <div className="pointer-events-none absolute left-3 top-3 z-20 max-w-[70%] rounded-md border border-white/20 bg-black/65 px-3 py-2 text-white">
          <p className="text-[11px] uppercase tracking-[0.14em] text-white/70">{currentLesson.course}</p>
          <p className="font-display text-sm font-semibold sm:text-base">{currentLesson.title}</p>
        </div>
        <div className="absolute right-3 top-3 z-20 rounded-md border border-white/20 bg-black/65 px-2 py-1 text-xs text-white">
          {isFocusMode ? "Focus mode" : "Lesson"} · {lessonIndex + 1}/{lessons.length}
        </div>
        <div className={cn("relative w-full", isFocusMode ? "h-full" : "aspect-video")}>
          {unavailable ? (
            <div className="flex h-full min-h-[240px] items-center justify-center bg-zinc-900 p-6 text-center text-sm text-zinc-200">
              This lesson could not be loaded. Check your connection and try the next lesson.
            </div>
          ) : (
            <iframe
              className="absolute inset-0 h-full w-full"
              src={`https://www.youtube.com/embed/${currentLesson.id}?rel=0&modestbranding=1`}
              title={currentLesson.title}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              onError={() => setUnavailable(true)}
            />
          )}
          {isFocusMode && (
            <div className="absolute inset-x-0 bottom-4 z-30 flex justify-center">
              <Button type="button" onClick={exitFocus}>
                <Unlock className="h-4 w-4" />
                Exit focus
              </Button>
            </div>
          )}
        </div>
      </div>

      {!isFocusMode && (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2" role="tablist" aria-label="Lessons">
            {lessons.map((lesson, index) => (
              <button
                key={lesson.id}
                type="button"
                role="tab"
                aria-selected={index === lessonIndex}
                aria-label={lesson.title}
                onClick={() => setLessonIndex(index)}
                className={cn(
                  "h-2.5 w-8 rounded-sm border-2 transition-colors",
                  index === lessonIndex
                    ? "border-foreground/30 bg-primary"
                    : "border-foreground/15 bg-muted hover:bg-muted-foreground/30"
                )}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" onClick={prevLesson} disabled={lessonIndex === 0}>
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={nextLesson}
              disabled={lessonIndex === lessons.length - 1}
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button type="button" onClick={enterFocus}>
              <Maximize className="h-4 w-4" />
              Focus mode
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
