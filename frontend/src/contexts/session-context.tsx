import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { API_BASE, NEGATIVE_EMOTIONS } from "@/lib/emotions";
import { lessons } from "@/lib/lessons";

export type DetectionStatus = "idle" | "initializing" | "active" | "error";
export type CameraDevice = { index: number; name: string; available?: boolean; selected?: boolean };

type SessionContextValue = {
  currentEmotion: string | null;
  detectionStatus: DetectionStatus;
  detectionSource: string;
  isAssistantOpen: boolean;
  isEmotionTriggered: boolean;
  isCameraOn: boolean;
  isMicOn: boolean;
  isBreathingOpen: boolean;
  isFocusMode: boolean;
  isDnd: boolean;
  isQuizOpen: boolean;
  lessonIndex: number;
  cameras: CameraDevice[];
  selectedCamera: string;
  currentLesson: (typeof lessons)[number];
  setIsCameraOn: (value: boolean) => void;
  setIsMicOn: (value: boolean) => void;
  setIsBreathingOpen: (value: boolean) => void;
  setIsFocusMode: (value: boolean) => void;
  setIsDnd: (value: boolean) => void;
  setIsQuizOpen: (value: boolean) => void;
  setLessonIndex: (value: number | ((prev: number) => number)) => void;
  handleEmotionChange: (emotion: string) => void;
  handleAssistantClose: () => void;
  handleManualAssistantToggle: () => void;
  handleCameraToggle: () => void;
  handleMicToggle: () => void;
  handleCameraSelect: (value: string) => Promise<void>;
  loadCameras: () => Promise<void>;
  nextLesson: () => void;
  prevLesson: () => void;
};

const SessionContext = createContext<SessionContextValue | null>(null);

async function notifyBackendSensorChange(sensorType: "camera" | "microphone", enabled: boolean) {
  try {
    const endpoint = sensorType === "camera" ? "/api/sensors/camera" : "/api/sensors/microphone";
    await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
  } catch (error) {
    console.error(`Failed to notify backend about ${sensorType} state:`, error);
  }
}

export const SessionProvider = ({ children }: { children: ReactNode }) => {
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [detectionStatus, setDetectionStatus] = useState<DetectionStatus>("idle");
  const [detectionSource, setDetectionSource] = useState("Sensors off");
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);
  const [isEmotionTriggered, setIsEmotionTriggered] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isMicOn, setIsMicOn] = useState(false);
  const [isBreathingOpen, setIsBreathingOpen] = useState(false);
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [isDnd, setIsDnd] = useState(false);
  const [isQuizOpen, setIsQuizOpen] = useState(false);
  const [lessonIndex, setLessonIndex] = useState(0);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState("0");

  const lastEmotionRef = useRef<string | null>(null);
  const assistantTriggeredRef = useRef(false);
  const isDndRef = useRef(isDnd);
  isDndRef.current = isDnd;

  const handleEmotionChange = useCallback((emotion: string) => {
    setCurrentEmotion(emotion);
    const isNegativeEmotion = NEGATIVE_EMOTIONS.includes(emotion);
    const isNewEmotion = emotion !== lastEmotionRef.current;

    if (
      isNegativeEmotion &&
      isNewEmotion &&
      !assistantTriggeredRef.current &&
      !isDndRef.current
    ) {
      assistantTriggeredRef.current = true;
      setIsAssistantOpen(false);
      setIsEmotionTriggered(false);
      setTimeout(() => {
        setIsEmotionTriggered(true);
        setIsAssistantOpen(true);
        setTimeout(() => {
          assistantTriggeredRef.current = false;
        }, 15000);
      }, 1000);
    }

    lastEmotionRef.current = emotion;
  }, []);

  const handleAssistantClose = useCallback(() => {
    setIsAssistantOpen(false);
    setIsEmotionTriggered(false);
    assistantTriggeredRef.current = false;
  }, []);

  const handleManualAssistantToggle = useCallback(() => {
    setIsEmotionTriggered(false);
    setIsAssistantOpen((open) => !open);
    assistantTriggeredRef.current = false;
  }, []);

  const handleMicToggle = useCallback(() => {
    setIsMicOn((prev) => {
      const next = !prev;
      notifyBackendSensorChange("microphone", next);
      return next;
    });
  }, []);

  const handleCameraToggle = useCallback(() => {
    setIsCameraOn((prev) => {
      const next = !prev;
      notifyBackendSensorChange("camera", next);
      return next;
    });
  }, []);

  const loadCameras = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sensors/cameras`);
      if (!res.ok) return;
      const data = await res.json();
      const list = Array.isArray(data.cameras) ? data.cameras : [];
      setCameras(list);
      const selected = data.selected_index ?? list.find((c: CameraDevice) => c.selected)?.index ?? 0;
      setSelectedCamera(String(selected));
    } catch (error) {
      console.error("Failed to list cameras:", error);
    }
  }, []);

  const handleCameraSelect = useCallback(
    async (value: string) => {
      setSelectedCamera(value);
      setIsCameraOn(true);
      try {
        await fetch(`${API_BASE}/api/sensors/camera/select`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: Number(value) }),
        });
        await loadCameras();
      } catch (error) {
        console.error("Failed to select camera:", error);
      }
    },
    [loadCameras]
  );

  const nextLesson = useCallback(() => {
    if (isFocusMode) return;
    setLessonIndex((index) => (index + 1) % lessons.length);
  }, [isFocusMode]);

  const prevLesson = useCallback(() => {
    if (isFocusMode) return;
    setLessonIndex((index) => (index - 1 + lessons.length) % lessons.length);
  }, [isFocusMode]);

  useEffect(() => {
    setIsCameraOn(true);
    notifyBackendSensorChange("camera", true);
    loadCameras();
  }, [loadCameras]);

  useEffect(() => {
    if (!isCameraOn && !isMicOn) {
      lastEmotionRef.current = null;
      setCurrentEmotion(null);
      setDetectionStatus("idle");
      setDetectionSource("Sensors off");
      return;
    }

    const source =
      isCameraOn && isMicOn
        ? "Combined facial and voice detection"
        : isCameraOn
          ? "Facial detection"
          : "Voice detection";
    setDetectionSource(source);
    setDetectionStatus("initializing");

    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/emotion/combined`);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        const detectedEmotion = data.emotion || "Unknown";
        setDetectionStatus("active");
        if (detectedEmotion !== lastEmotionRef.current) {
          handleEmotionChange(detectedEmotion);
        }
      } catch (error) {
        console.error("Emotion poll failed:", error);
        try {
          const jsonRes = await fetch(`${API_BASE}/api/emotion/latest`);
          if (jsonRes.ok) {
            const jsonData = await jsonRes.json();
            const fallbackEmotion = jsonData.final_emotion || "Unknown";
            setDetectionStatus("active");
            if (fallbackEmotion !== lastEmotionRef.current) {
              handleEmotionChange(fallbackEmotion);
            }
            return;
          }
        } catch {
          /* fall through */
        }
        setDetectionStatus("error");
        const unknown = "Unknown";
        if (unknown !== lastEmotionRef.current) {
          handleEmotionChange(unknown);
        }
      }
    };

    tick();
    const interval = setInterval(tick, 2000);
    return () => clearInterval(interval);
  }, [handleEmotionChange, isCameraOn, isMicOn]);

  const value = useMemo<SessionContextValue>(
    () => ({
      currentEmotion,
      detectionStatus,
      detectionSource,
      isAssistantOpen,
      isEmotionTriggered,
      isCameraOn,
      isMicOn,
      isBreathingOpen,
      isFocusMode,
      isDnd,
      isQuizOpen,
      lessonIndex,
      cameras,
      selectedCamera,
      currentLesson: lessons[lessonIndex],
      setIsCameraOn,
      setIsMicOn,
      setIsBreathingOpen,
      setIsFocusMode,
      setIsDnd,
      setIsQuizOpen,
      setLessonIndex,
      handleEmotionChange,
      handleAssistantClose,
      handleManualAssistantToggle,
      handleCameraToggle,
      handleMicToggle,
      handleCameraSelect,
      loadCameras,
      nextLesson,
      prevLesson,
    }),
    [
      cameras,
      currentEmotion,
      detectionSource,
      detectionStatus,
      handleAssistantClose,
      handleCameraSelect,
      handleCameraToggle,
      handleEmotionChange,
      handleManualAssistantToggle,
      handleMicToggle,
      isAssistantOpen,
      isBreathingOpen,
      isCameraOn,
      isDnd,
      isEmotionTriggered,
      isFocusMode,
      isMicOn,
      isQuizOpen,
      lessonIndex,
      loadCameras,
      nextLesson,
      prevLesson,
      selectedCamera,
    ]
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
};

export const useSession = () => {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useSession must be used within SessionProvider");
  }
  return context;
};
