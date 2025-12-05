import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

import {
  Camera,
  CameraOff,
  Mic,
  MicOff,
  Coffee,
  MessageCircle,
  BookOpen,
  Timer,
} from "lucide-react";

import "./voice.css"; // Optional styling for voice UI

interface ControlPanelProps {
  isCameraOn: boolean;
  setIsCameraOn: (value: boolean) => void;
  isMicOn: boolean;
  setIsMicOn: (value: boolean) => void;
  onBreathingExercise: () => void;
  onAssistantToggle: () => void;
  onEmotionChange: (emotion: string) => void; // New prop to pass emotion to parent
}

export const ControlPanel = ({
  isCameraOn,
  setIsCameraOn,
  isMicOn,
  setIsMicOn,
  onBreathingExercise,
  onAssistantToggle,
  onEmotionChange,
}: ControlPanelProps) => {
  const [showVoiceInput, setShowVoiceInput] = useState(false);
  const [showAssistant, setShowAssistant] = useState(false);
  const [emotion, setEmotion] = useState<string | null>(null);
  const [lastEmotion, setLastEmotion] = useState<string | null>(null);

  // Function to notify backend about sensor state changes
  const notifyBackendSensorChange = async (sensorType: 'camera' | 'microphone', enabled: boolean) => {
    try {
      const endpoint = sensorType === 'camera' ? '/api/sensors/camera' : '/api/sensors/microphone';
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log(`Backend ${sensorType} state updated:`, data);
      }
    } catch (error) {
      console.error(`Failed to notify backend about ${sensorType} state:`, error);
    }
  };

  const handleMicToggle = () => {
    const newMicState = !isMicOn;
    setIsMicOn(newMicState);
    setShowVoiceInput(newMicState);
    
    // Notify backend about microphone state change
    notifyBackendSensorChange('microphone', newMicState);
  };

  const handleCameraToggle = () => {
    const newCameraState = !isCameraOn;
    
    // Don't access camera directly from frontend - let backend handle it
    setIsCameraOn(newCameraState);
    
    // Notify backend about camera state change
    notifyBackendSensorChange('camera', newCameraState);
  };

  const handleVoiceResult = (text: string) => {
    console.log("Recognized speech:", text);
  };

  const handleAssistantClick = () => {
    setShowAssistant((prev) => !prev);
    onAssistantToggle();
  };

  // Function to get emotion based on current state
  const getEmotionBasedOnState = async () => {
    try {
      let endpoint = "";
      
      if (!isCameraOn && !isMicOn) {
        // Both off - use interaction emotion
        endpoint = "/api/emotion/interaction";
      } else if (isCameraOn && !isMicOn) {
        // Only camera on - use facial emotion
        endpoint = "/api/emotion/facial";
      } else if (!isCameraOn && isMicOn) {
        // Only mic on - use voice emotion
        endpoint = "/api/emotion/voice";
      } else {
        // Both on - use combined emotion
        endpoint = "/api/emotion/combined";
      }

      const res = await fetch(`http://localhost:8000${endpoint}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      const detectedEmotion = data.emotion || "Unknown";
      
      console.log(`Fetched emotion from ${endpoint}:`, detectedEmotion);
      
      // Only update if emotion has changed
      if (detectedEmotion !== lastEmotion) {
        setEmotion(detectedEmotion);
        setLastEmotion(detectedEmotion);
        onEmotionChange(detectedEmotion); // Pass to parent component
        
        console.log(`Emotion changed: ${detectedEmotion} (${endpoint})`);
      }
      
      return detectedEmotion;
    } catch (error) {
      console.error("Emotion fetch failed:", error);
      
      // Try to get latest emotion from JSON file as fallback
      try {
        const jsonRes = await fetch('http://localhost:8000/api/emotion/latest');
        if (jsonRes.ok) {
          const jsonData = await jsonRes.json();
          const fallbackEmotion = jsonData.final_emotion || "Unknown";
          console.log("Using fallback emotion from JSON:", fallbackEmotion);
          
          if (fallbackEmotion !== lastEmotion) {
            setEmotion(fallbackEmotion);
            setLastEmotion(fallbackEmotion);
            onEmotionChange(fallbackEmotion);
          }
          return fallbackEmotion;
        }
      } catch (jsonError) {
        console.error("JSON fallback also failed:", jsonError);
      }
      
      const fallbackEmotion = "Unknown";
      if (fallbackEmotion !== lastEmotion) {
        setEmotion(fallbackEmotion);
        setLastEmotion(fallbackEmotion);
        onEmotionChange(fallbackEmotion);
      }
      return fallbackEmotion;
    }
  };

  // Poll emotion from backend based on current state
  useEffect(() => {
    // Only poll if at least one sensor is enabled
    if (!isCameraOn && !isMicOn) {
      console.log("No sensors enabled, skipping emotion polling");
      return;
    }

    const fetchEmotion = async () => {
      await getEmotionBasedOnState();
    };

    // Initial fetch
    fetchEmotion();
    
    // Set up polling interval
    const interval = setInterval(fetchEmotion, 2000); // Poll every 2 seconds for more responsive updates

    return () => clearInterval(interval);
  }, [isCameraOn, isMicOn]); // Re-run when camera or mic state changes

  // Function to check backend status
  const checkBackendStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sensors/status');
      if (res.ok) {
        const status = await res.json();
        console.log('Backend status:', status);
        return status;
      }
    } catch (error) {
      console.error('Failed to check backend status:', error);
    }
    return null;
  };

  // Check backend status when sensors are toggled
  useEffect(() => {
    if (isCameraOn || isMicOn) {
      checkBackendStatus();
    }
  }, [isCameraOn, isMicOn]);

  return (
    <Card className="p-6 bg-white/60 backdrop-blur-sm border-0 shadow-lg">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-800">Learning Controls</h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Camera Toggle */}
          <Button
            variant={isCameraOn ? "default" : "outline"}
            className={`flex flex-col items-center space-y-2 h-20 ${
              isCameraOn
                ? "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
                : "border-2 border-gray-200 hover:border-blue-300"
            }`}
            onClick={handleCameraToggle}
          >
            {isCameraOn ? <Camera className="w-5 h-5" /> : <CameraOff className="w-5 h-5" />}
            <span className="text-xs">Camera</span>
          </Button>

          {/* Mic Toggle */}
          <Button
            variant={isMicOn ? "default" : "outline"}
            className={`flex flex-col items-center space-y-2 h-20 ${
              isMicOn
                ? "bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700"
                : "border-2 border-gray-200 hover:border-green-300"
            }`}
            onClick={handleMicToggle}
          >
            {isMicOn ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
            <span className="text-xs">Microphone</span>
          </Button>

          {/* Break Timer */}
          <Button
            variant="outline"
            className="flex flex-col items-center space-y-2 h-20 border-2 border-gray-200 hover:border-orange-300 hover:bg-orange-50"
            onClick={onBreathingExercise}
          >
            <Coffee className="w-5 h-5 text-orange-500" />
            <span className="text-xs">Take Break</span>
          </Button>

          {/* AI Assistant */}
          <Button
            variant="outline"
            className="flex flex-col items-center space-y-2 h-20 border-2 border-gray-200 hover:border-purple-300 hover:bg-purple-50"
            onClick={handleAssistantClick}
          >
            <MessageCircle className="w-5 h-5 text-purple-500" />
            <span className="text-xs">AI Help</span>
          </Button>
        </div>

        {/* Camera Status Indicator */}
        {isCameraOn && (
          <div className="mt-4 p-3 bg-blue-100 rounded-lg shadow text-sm font-medium text-gray-800">
            📹 <strong>Camera Status:</strong> Active (Backend Processing)
            <div className="text-xs text-gray-600 mt-1">
              Camera is being used by the emotion detection system
            </div>
          </div>
        )}

        {/* Current Emotion */}
        {emotion && (
          <div className="mt-4 p-3 bg-yellow-100 rounded-lg shadow text-sm font-medium text-gray-800">
            🎭 <strong>Current Mood:</strong> {emotion}
            <div className="text-xs text-gray-600 mt-1">
              {!isCameraOn && !isMicOn && "Based on interaction"}
              {isCameraOn && !isMicOn && "Based on facial detection"}
              {!isCameraOn && isMicOn && "Based on voice detection"}
              {isCameraOn && isMicOn && "Based on combined detection"}
            </div>
            <div className="text-xs text-green-600 mt-1">
              ✓ Live emotion detection active
            </div>
          </div>
        )}

        {/* Status when no emotion detected */}
        {!emotion && (isCameraOn || isMicOn) && (
          <div className="mt-4 p-3 bg-blue-100 rounded-lg shadow text-sm font-medium text-gray-800">
            🔄 <strong>Status:</strong> Initializing emotion detection...
            <div className="text-xs text-gray-600 mt-1">
              {isCameraOn && "Camera active"}
              {isMicOn && "Microphone active"}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2 pt-4 border-t border-gray-200">
          <Button size="sm" variant="outline" className="hover:bg-blue-50 hover:border-blue-300">
            <BookOpen className="w-4 h-4 mr-2" />
            Quiz Me
          </Button>
          <Button size="sm" variant="outline" className="hover:bg-green-50 hover:border-green-300">
            <Timer className="w-4 h-4 mr-2" />
            Focus Mode
          </Button>
          <Button size="sm" variant="outline" className="hover:bg-red-50 hover:border-red-300">
            Don't Disturb
          </Button>
        </div>
      </div>
    </Card>
  );
};
