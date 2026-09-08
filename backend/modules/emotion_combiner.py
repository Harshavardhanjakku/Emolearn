from modules.facial_emotion import (
    get_facial_emotion,
    start_facial_emotion_detection,
    stop_facial_emotion_detection,
    set_camera_session_allowed,
)
from modules.speech_emotion import get_speech_emotion, start_speech_emotion_detection, stop_speech_emotion_detection
from modules.mouse_emotion import (
    get_mouse_emotion,
    start_mouse_emotion_detection,
    stop_mouse_emotion_detection,
    has_mouse_hook_input,
)
from modules.emotion_log_store import append_entry as append_emotion_log_entry

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track which detectors are running
detectors_running = {
    "facial": False,
    "speech": False,
    "mouse": False
}

# Highest rank wins — stops "always Engaged" when face says Engaged but mouse/voice shows disengagement.
CONCERN_RANK = {
    "Frustrated": 6,
    "Face Not Detected": 5,
    "Confused": 4,
    "Sleepy": 3,
    "Bored": 3,
    "Happy": 2,
    "Engaged": 1,
    "Unknown": 0,
}


def merge_multimodal_emotion(facial: str, speech: str, mouse: str) -> str:
    pool = [
        e
        for e in (facial, speech, mouse)
        if e and str(e).strip() and e != "Unknown"
    ]
    if not pool:
        return "Engaged"
    return max(pool, key=lambda e: CONCERN_RANK.get(e, 2))


def start_detector(detector_type: str):
    """Start a specific detector"""
    global detectors_running
    try:
        if detector_type == "facial" and not detectors_running["facial"]:
            start_facial_emotion_detection()
            detectors_running["facial"] = True
            logger.info("Facial emotion detection started")
        elif detector_type == "speech" and not detectors_running["speech"]:
            start_speech_emotion_detection()
            detectors_running["speech"] = True
            logger.info("Speech emotion detection started")
        elif detector_type == "mouse" and not detectors_running["mouse"]:
            start_mouse_emotion_detection()
            detectors_running["mouse"] = True
            logger.info("Mouse emotion detection started")
    except Exception as e:
        logger.error(f"Failed to start {detector_type} detector: {e}")

def stop_detector(detector_type: str):
    """Stop a specific detector"""
    global detectors_running
    try:
        if detector_type == "facial" and detectors_running["facial"]:
            stop_facial_emotion_detection()
            detectors_running["facial"] = False
            logger.info("Facial emotion detection stopped")
        elif detector_type == "speech" and detectors_running["speech"]:
            stop_speech_emotion_detection()
            detectors_running["speech"] = False
            logger.info("Speech emotion detection stopped")
        elif detector_type == "mouse" and detectors_running["mouse"]:
            stop_mouse_emotion_detection()
            detectors_running["mouse"] = False
            logger.info("Mouse emotion detection stopped")
    except Exception as e:
        logger.error(f"Failed to stop {detector_type} detector: {e}")

def update_detectors(camera_enabled: bool, microphone_enabled: bool):
    """Update which detectors should be running based on sensor states"""
    logger.info(f"Updating detectors - Camera: {camera_enabled}, Microphone: {microphone_enabled}")
    logger.info(f"Current detector states: {detectors_running}")
    
    # Always keep mouse detection running for interaction-based emotion
    if not detectors_running["mouse"]:
        logger.info("Starting mouse emotion detection")
        start_detector("mouse")
    
    # Start/stop facial detection based on camera state
    if camera_enabled and not detectors_running["facial"]:
        logger.info("Starting facial emotion detection")
        set_camera_session_allowed(True)
        start_detector("facial")
    elif not camera_enabled and detectors_running["facial"]:
        logger.info("Stopping facial emotion detection")
        set_camera_session_allowed(False)
        stop_detector("facial")
    
    # Start/stop speech detection based on microphone state
    if microphone_enabled and not detectors_running["speech"]:
        logger.info("Starting speech emotion detection")
        start_detector("speech")
    elif not microphone_enabled and detectors_running["speech"]:
        logger.info("Stopping speech emotion detection")
        stop_detector("speech")
    
    logger.info(f"Updated detector states: {detectors_running}")

def get_combined_emotion():
    try:
        # Get emotions from running detectors only
        facial = get_facial_emotion() if detectors_running["facial"] else "Unknown"
        speech = get_speech_emotion() if detectors_running["speech"] else "Unknown"
        mouse = (
            get_mouse_emotion()
            if detectors_running["mouse"] and has_mouse_hook_input()
            else "Unknown"
        )

        # Debug logging to see what each detector is returning
        logger.info(f"DEBUG - Facial emotion: {facial}")
        logger.info(f"DEBUG - Speech emotion: {speech}")
        logger.info(f"DEBUG - Mouse emotion: {mouse}")

        final = merge_multimodal_emotion(facial, speech, mouse)

        # Emotion weight map (optional fine-tuning)
        emotion_weights = {
            "Bored": 0.45,
            "Confused": 0.6,
            "Frustrated": 0.75,
            "Sleepy": 0.5,
            "Face Not Detected": 0.25,
            "Engaged": 0.9,
            "Unknown": 0.2
        }
        value = emotion_weights.get(final, 0.0)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            "time": now,
            "facial_emotion": facial,
            "voice_emotion": speech,
            "interaction_emotion": mouse,
            "final_emotion": final,
            "value": value
        }

        try:
            append_emotion_log_entry(log_entry)
        except Exception as e:
            logger.error(f"Error writing to emotion_log.json: {e}")

        return final

    except Exception as e:
        logger.error(f"Error in get_combined_emotion: {e}")
        return "Engaged"

if __name__ == "__main__":
    print("🧠 Starting Combined Emotion Check...")
    try:
        while True:
            emotion = get_combined_emotion()
            print(f"[Combined Emotion]: {emotion}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("🛑 Stopping combined emotion check.")