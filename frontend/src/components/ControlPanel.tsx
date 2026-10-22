import {
  BookOpen,
  Camera,
  CameraOff,
  Coffee,
  MessageCircle,
  Mic,
  MicOff,
  RefreshCw,
  ShieldAlert,
  Timer,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSession } from "@/contexts/session-context";
import { cn } from "@/lib/utils";
import "./voice.css";

const ControlButton = ({
  active,
  label,
  onClick,
  children,
}: {
  active?: boolean;
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) => (
  <Button
    type="button"
    variant={active ? "default" : "outline"}
    className="h-auto min-h-16 flex-col gap-1 py-3"
    aria-pressed={!!active}
    onClick={onClick}
  >
    {children}
    <span className="text-xs">{label}</span>
  </Button>
);

export const ControlPanel = ({ compact = false }: { compact?: boolean }) => {
  const {
    isCameraOn,
    isMicOn,
    isFocusMode,
    isDnd,
    currentEmotion,
    detectionStatus,
    detectionSource,
    cameras,
    selectedCamera,
    handleCameraToggle,
    handleMicToggle,
    handleCameraSelect,
    handleManualAssistantToggle,
    loadCameras,
    setIsBreathingOpen,
    setIsFocusMode,
    setIsDnd,
    setIsQuizOpen,
  } = useSession();

  const selectedName = cameras.find((c) => String(c.index) === selectedCamera)?.name;
  const selectedUnavailable = cameras.find((c) => String(c.index) === selectedCamera)?.available === false;

  return (
    <Card className="p-5">
      <div className="space-y-5">
        <div>
          <p className="page-kicker">Session</p>
          <h3 className="font-display text-lg font-bold">Learning controls</h3>
        </div>

        <section className="space-y-3" aria-labelledby="sensor-controls">
          <h4 id="sensor-controls" className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            Sensors
          </h4>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <ControlButton active={isCameraOn} label={isCameraOn ? "Camera on" : "Camera off"} onClick={handleCameraToggle}>
              {isCameraOn ? <Camera className="h-5 w-5" /> : <CameraOff className="h-5 w-5" />}
            </ControlButton>
            <ControlButton active={isMicOn} label={isMicOn ? "Mic on" : "Mic off"} onClick={handleMicToggle}>
              {isMicOn ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
            </ControlButton>
            <ControlButton label="Take break" onClick={() => setIsBreathingOpen(true)}>
              <Coffee className="h-5 w-5" />
            </ControlButton>
            <ControlButton label="AI help" onClick={handleManualAssistantToggle}>
              <MessageCircle className="h-5 w-5" />
            </ControlButton>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs font-medium" htmlFor="camera-device">
                Camera device
              </label>
              <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={loadCameras}>
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh
              </Button>
            </div>
            <Select value={selectedCamera} onValueChange={handleCameraSelect}>
              <SelectTrigger id="camera-device" className="bg-card">
                <SelectValue placeholder="Choose a camera" />
              </SelectTrigger>
              <SelectContent>
                {cameras.length === 0 ? (
                  <SelectItem value="0">Camera 0 (default)</SelectItem>
                ) : (
                  cameras.map((cam) => (
                    <SelectItem key={cam.index} value={String(cam.index)}>
                      {cam.name}
                      {cam.available === false ? " — not responding" : ""}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
            {selectedUnavailable && (
              <p className="text-xs text-destructive">Selected camera is not responding. Try another device.</p>
            )}
          </div>

          <div
            className={cn(
              "rounded-md border-2 px-3 py-2 text-sm",
              detectionStatus === "error"
                ? "border-destructive/40 bg-destructive/10"
                : "border-border bg-muted/40"
            )}
          >
            <p>
              <span className="font-semibold">Camera:</span>{" "}
              {isCameraOn
                ? `On · ${selectedName || "default camera"}`
                : "Off · emotion detection needs a sensor"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {detectionStatus === "idle" && "Emotion detection is off."}
              {detectionStatus === "initializing" && "Connecting to the detection service…"}
              {detectionStatus === "active" && `Detection running · ${detectionSource}`}
              {detectionStatus === "error" && "Detection service unreachable. Status is unknown, not live."}
            </p>
            {currentEmotion && detectionStatus === "active" && (
              <p className="mt-1 text-xs">
                Current mood: <span className="font-semibold">{currentEmotion}</span>
              </p>
            )}
          </div>
        </section>

        {!compact && (
          <section className="space-y-3 border-t-2 border-border pt-4" aria-labelledby="study-controls">
            <h4 id="study-controls" className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              Study
            </h4>
            <div className="flex flex-wrap gap-2">
              <Button type="button" size="sm" variant="outline" onClick={() => setIsQuizOpen(true)}>
                <BookOpen className="h-4 w-4" />
                Quiz me
              </Button>
              <Button
                type="button"
                size="sm"
                variant={isFocusMode ? "default" : "outline"}
                aria-pressed={isFocusMode}
                onClick={() => setIsFocusMode(!isFocusMode)}
              >
                <Timer className="h-4 w-4" />
                {isFocusMode ? "Focus on" : "Focus mode"}
              </Button>
              <Button
                type="button"
                size="sm"
                variant={isDnd ? "default" : "outline"}
                aria-pressed={isDnd}
                onClick={() => setIsDnd(!isDnd)}
              >
                <ShieldAlert className="h-4 w-4" />
                {isDnd ? "Do not disturb on" : "Do not disturb"}
              </Button>
            </div>
          </section>
        )}
      </div>
    </Card>
  );
};
