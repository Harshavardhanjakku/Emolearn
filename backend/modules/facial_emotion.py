import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import time
import pandas as pd
from datetime import datetime
import threading
import logging
import os
from collections import deque
import traceback

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging to match other modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Shared State and Constants ---
STATE = {
    "running": False,
    "last_emotion": "Unknown",
    "lock": threading.Lock(),
    "log_data": [],
    "emotion_history": deque(maxlen=3),
    "camera_available": False,
    "last_camera_check": 0,
    "reopen_camera": False,
}

# Emotion index mapping (keep as is)
emotion_index = {"Engaged": 0, "Confused": 1, "Frustrated": 2, "Bored": 3, "Sleepy": 4, "Unknown": 5}

# Constants for emotion detection thresholds (tuned for less sensitivity)
# DeepFace returns emotion dict values on 0-100 (%). Older code used thresholds like 0.40 vs that scale,
# so almost any non-zero angry score beat the threshold → constant "Frustrated".
FRUSTRATION_PROB_THRESHOLD = 48.0  # Min % for max(angry, disgust) vs calm_peak + margin below
FRUSTRATION_MARGIN_OVER_CALM = 10.0  # stress must exceed best(neutral,happy) by this many points (0-100)
FRUSTRATION_EXTREME = 62.0  # strong anger/disgust signal (0-100)
EAR_THRESHOLD_SLEEPY = 0.20       # Slightly increased for less sensitivity to minor blinks
MAR_THRESHOLD_YAWN = 0.45         # Slightly increased for less sensitivity to minor mouth movements
PITCH_NEUTRAL_RANGE = (0.00, 0.15)  # Keep as is for now
EYES_CLOSED_DURATION_SLEEPY = 0.35
YAWN_DURATION_BORED = 0.6
NO_FACE_DURATION_BORED = 2.0
GAZE_AWAY_THRESHOLD = 45          # Increased from 30 for less sensitivity to gaze shifts
HEAD_TILT_THRESHOLD = 0.12        # Increased from 0.1 for less sensitivity to head tilts
EMOTION_LOCK_DURATION = 0
CAMERA_RETRY_INTERVAL = 1

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5
)
cap = None
CAMERA_INDEX = 0
_CAMERA_INDEX_LOCK = threading.Lock()

_deepface_frame_lock = threading.Lock()
_deepface_latest_frame = None
_deepface_result = "Unknown"
_deepface_worker_started = False


def _map_deepface_result(deepface_result):
    dom_emotion = deepface_result[0]["dominant_emotion"]
    emotion_probabilities = deepface_result[0]["emotion"]
    angry_p = emotion_probabilities.get("angry", 0)
    disgust_p = emotion_probabilities.get("disgust", 0)
    neutral_p = emotion_probabilities.get("neutral", 0)
    happy_p = emotion_probabilities.get("happy", 0)
    stress = max(angry_p, disgust_p)
    calm_peak = max(neutral_p, happy_p)
    stressed_enough_vs_calm = (
        stress >= FRUSTRATION_PROB_THRESHOLD
        and stress >= calm_peak + FRUSTRATION_MARGIN_OVER_CALM
    )
    extreme_stress = stress >= FRUSTRATION_EXTREME and stress >= calm_peak + 5.0
    if stressed_enough_vs_calm or extreme_stress:
        return "Frustrated"
    if dom_emotion in ["neutral", "happy"]:
        return "Engaged"
    if dom_emotion in ["sad", "fear"]:
        return "Confused"
    return "Unknown"


def _deepface_worker():
    global _deepface_latest_frame, _deepface_result
    while True:
        if not CAMERA_SESSION_ALLOWED.is_set():
            time.sleep(0.05)
            continue
        with _deepface_frame_lock:
            frame = _deepface_latest_frame
            _deepface_latest_frame = None
        if frame is None:
            time.sleep(0.01)
            continue
        try:
            deepface_result = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False, silent=True
            )
            _deepface_result = _map_deepface_result(deepface_result)
        except Exception:
            pass


def _ensure_deepface_worker():
    global _deepface_worker_started
    if not _deepface_worker_started:
        _deepface_worker_started = True
        threading.Thread(target=_deepface_worker, daemon=True).start()


def _submit_frame_for_deepface(frame):
    global _deepface_latest_frame
    with _deepface_frame_lock:
        _deepface_latest_frame = frame.copy()

# When False, never open VideoCapture (avoids webcam LED / snapshots after user turns camera off).
CAMERA_SESSION_ALLOWED = threading.Event()


def set_camera_session_allowed(allowed: bool) -> None:
    """Call from emotion_combiner when facial detection should/should not touch the webcam."""
    if allowed:
        CAMERA_SESSION_ALLOWED.set()
        STATE["last_camera_check"] = 0  # next availability check probes hardware again
    else:
        CAMERA_SESSION_ALLOWED.clear()


# --- Camera Management Functions ---

def check_camera_availability():
    """True if this process may use the webcam. Do not open a second capture (that flickers the LED)."""
    try:
        if not CAMERA_SESSION_ALLOWED.is_set():
            return False
        if cap is not None and cap.isOpened():
            STATE["camera_available"] = True
            return True
        STATE["camera_available"] = True
        return True
    except Exception as e:
        logger.debug(f"Camera availability check failed: {str(e)}")
        STATE["camera_available"] = False
        return False

def get_camera_index() -> int:
    with _CAMERA_INDEX_LOCK:
        return CAMERA_INDEX


def _camera_label(index: int) -> str:
    if index == 0:
        return "Camera 0 (built-in / default)"
    return f"Camera {index} (external)"


def _try_open_index(index: int):
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)
    for backend in backends:
        test = cv2.VideoCapture(index, backend)
        if test.isOpened():
            ok, _ = test.read()
            if ok:
                return test
        test.release()
    return None


def list_available_cameras(max_index: int = 6):
    """Probe camera indexes so the UI can pick built-in or USB webcams."""
    found = []
    selected = get_camera_index()
    active_open = cap is not None and cap.isOpened()
    for i in range(max_index):
        if active_open and i == selected:
            found.append({"index": i, "name": _camera_label(i), "available": True, "selected": True})
            continue
        test = _try_open_index(i)
        if test is not None:
            test.release()
            found.append({"index": i, "name": _camera_label(i), "available": True, "selected": i == selected})
    if not any(c["index"] == selected for c in found):
        found.insert(0, {"index": selected, "name": _camera_label(selected), "available": False, "selected": True})
    if not found:
        found.append({"index": 0, "name": _camera_label(0), "available": False, "selected": True})
    return found


def set_camera_index(index: int, reopen: bool = True) -> int:
    """Switch to another webcam (USB/external) and reopen if the loop is live."""
    global CAMERA_INDEX
    with _CAMERA_INDEX_LOCK:
        CAMERA_INDEX = int(index)
        chosen = CAMERA_INDEX
    if reopen:
        with STATE["lock"]:
            STATE["reopen_camera"] = True
    logger.info("Camera device set to index %s", chosen)
    return chosen


def open_best_camera():
    """Open the selected camera; if it fails, try other connected cameras."""
    preferred = get_camera_index()
    order = [preferred] + [i for i in range(6) if i != preferred]
    for i in order:
        opened = _try_open_index(i)
        if opened is not None:
            if i != preferred:
                set_camera_index(i, reopen=False)
                logger.info("Selected camera %s failed; using camera %s", preferred, i)
            opened.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            opened.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            opened.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            STATE["camera_available"] = True
            return opened
    STATE["camera_available"] = False
    return None


def safe_camera_release():
    """Safely release camera resources."""
    global cap
    if cap is not None:
        try:
            cap.release()
            cap = None
        except Exception as e:
            logger.debug(f"Error releasing camera: {str(e)}")

# --- Facial Metric Calculation Functions ---

def calculate_ear(points, landmarks, w, h):
    """Calculate Eye Aspect Ratio (EAR) for an eye."""
    try:
        coords = [(int(landmarks.landmark[p].x * w), int(landmarks.landmark[p].y * h)) for p in points]
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        return (A + B) / (2.0 * C) if C != 0 else 0
    except Exception as e:
        logger.error(f"Error calculating EAR: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0

def calculate_mar(points, landmarks, w, h):
    """Calculate Mouth Aspect Ratio (MAR)."""
    try:
        coords = [(int(landmarks.landmark[p].x * w), int(landmarks.landmark[p].y * h)) for p in points]
        A = np.linalg.norm(np.array(coords[2]) - np.array(coords[6]))
        B = np.linalg.norm(np.array(coords[0]) - np.array(coords[4]))
        return A / B if B != 0 else 0
    except Exception as e:
        logger.error(f"Error calculating MAR: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0

def calculate_pitch(landmarks):
    """Calculate head pitch (up/down rotation).
    Using nose tip (1) and chin (152) for more robust pitch estimation.
    A positive value generally means looking down, negative means looking up.
    """
    try:
        # Using normalized coordinates for pitch calculation
        nose_tip_y = landmarks.landmark[1].y
        chin_y = landmarks.landmark[152].y
        # A simple difference can indicate pitch. Adjust sensitivity as needed.
        return nose_tip_y - chin_y
    except Exception as e:
        logger.error(f"Error calculating pitch: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0

def is_head_tilted(landmarks):
    """Check if head is tilted left or right (roll)."""
    try:
        # Using left (234) and right (454) temples or ears for roll detection
        # Increased threshold for less sensitivity
        return abs(landmarks.landmark[234].y - landmarks.landmark[454].y) > HEAD_TILT_THRESHOLD
    except Exception as e:
        logger.error(f"Error checking head tilt: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

# --- Calibration and Smoothing Functions ---

def calibrate():
    """Neutral eye baseline; filled in-stream so the camera stays open."""
    return 0.5

def get_majority_emotion(history_deque):
    """Return the most frequent emotion in the history deque."""
    try:
        if not history_deque:
            return "Unknown"
        counts = {}
        for emotion in history_deque:
            counts[emotion] = counts.get(emotion, 0) + 1
        return max(counts, key=counts.get)
    except Exception as e:
        logger.error(f"Error getting majority emotion: {str(e)}")
        logger.debug(traceback.format_exc())
        return "Unknown"

# --- Main Emotion Detection Loop ---

def facial_emotion_loop():
    """Core loop for facial emotion detection."""
    global cap
    my_id = threading.get_ident()
    loop_cap = None
    with STATE["lock"]:
        STATE["loop_ident"] = my_id
    try:
        _ensure_deepface_worker()
        calibrated_eye_y = calibrate()

        if not CAMERA_SESSION_ALLOWED.is_set() or not STATE["running"]:
            logger.info("Camera session stopped — exiting facial loop.")
            return

        cap = open_best_camera()
        loop_cap = cap
        if cap is None or not cap.isOpened():
            raise Exception("Failed to open any webcam. Plug in an external camera and pick it in the UI.")
        gaze_away_counter = 0
        eyes_closed_start = None
        yawn_start = None
        no_face_start = None
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while STATE["running"]:
            if not CAMERA_SESSION_ALLOWED.is_set():
                break
            try:
                with STATE["lock"]:
                    need_reopen = STATE.get("reopen_camera")
                    if need_reopen:
                        STATE["reopen_camera"] = False
                if need_reopen or cap is None or not cap.isOpened():
                    if cap is loop_cap:
                        safe_camera_release()
                    cap = open_best_camera()
                    loop_cap = cap
                    if cap is None or not cap.isOpened():
                        time.sleep(0.4)
                        continue
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        logger.warning("Too many consecutive frame failures, checking camera availability")
                        if not check_camera_availability():
                            logger.warning("Camera not available, using fallback emotion")
                            time.sleep(0.2)
                            continue
                        consecutive_failures = 0
                    else:
                        logger.debug("Failed to read frame from webcam, retrying...")
                        time.sleep(0.1)  # Reduced sleep time
                        continue

                consecutive_failures = 0  # Reset failure counter on success
                _submit_frame_for_deepface(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                h, w, _ = frame.shape
                current_emotion = "Unknown"  # Default for this frame

                if results.multi_face_landmarks:
                    no_face_start = None;  # Reset no face timer
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye_landmarks = [33, 160, 158, 133, 153, 144]
                        right_eye_landmarks = [362, 385, 387, 263, 373, 380]
                        mouth_landmarks = [61, 291, 0, 17, 405, 185, 13]

                        ear = (calculate_ear(left_eye_landmarks, face_landmarks, w, h) +
                               calculate_ear(right_eye_landmarks, face_landmarks, w, h)) / 2
                        mar = calculate_mar(mouth_landmarks, face_landmarks, w, h)
                        # Recalculate eye_y based on the current frame's landmarks
                        eye_y = (face_landmarks.landmark[33].y + face_landmarks.landmark[263].y) / 2.0
                        pitch = calculate_pitch(face_landmarks)

                        # --- Emotion Detection Logic (Prioritized) ---

                        # 1. Sleepy
                        if ear < EAR_THRESHOLD_SLEEPY:
                            if eyes_closed_start is None:
                                eyes_closed_start = time.time()
                            elif time.time() - eyes_closed_start > EYES_CLOSED_DURATION_SLEEPY:
                                current_emotion = "Sleepy"
                                # If sleepy, reset other counters
                                yawn_start = None
                                gaze_away_counter = 0
                        else:
                            eyes_closed_start = None  # Reset eyes closed timer

                        # 2. Bored (Yawning or Gaze Away)
                        if current_emotion != "Sleepy":  # Only check for bored if not already sleepy
                            if mar > MAR_THRESHOLD_YAWN:  # and pitch < PITCH_NEUTRAL_RANGE[1]: Removed pitch constraint for yawning, focus on mouth
                                if yawn_start is None:
                                    yawn_start = time.time()
                                elif time.time() - yawn_start > YAWN_DURATION_BORED:
                                    current_emotion = "Bored"
                                    eyes_closed_start = None  # Reset sleepy timer if bored by yawning
                            else:
                                yawn_start = None

                            # Gaze Away for Boredom
                            # Only if not already identified as sleepy or yawning
                            if current_emotion not in ["Sleepy", "Bored"]:
                                if abs(eye_y - calibrated_eye_y) > 0.07:  # Keeping original gaze deviation logic
                                    gaze_away_counter += 1
                                else:
                                    gaze_away_counter = max(0, gaze_away_counter - 2)  # Decays faster

                                if gaze_away_counter > GAZE_AWAY_THRESHOLD:
                                    current_emotion = "Bored"
                                    gaze_away_counter = 0  # Reset counter once bored is detected

                        # 3. Confused (Head Tilt)
                        if current_emotion not in ["Sleepy", "Bored"]:  # Check for confused if not sleepy or bored
                            if is_head_tilted(face_landmarks):
                                current_emotion = "Confused"
                                eyes_closed_start = None
                                yawn_start = None
                                gaze_away_counter = 0

                        # 4. DeepFace (async, latest frame) — MediaPipe face cues already won if set
                        if current_emotion == "Unknown":
                            df_emotion = _deepface_result
                            if df_emotion != "Unknown":
                                current_emotion = df_emotion

                        if current_emotion == "Unknown":
                            current_emotion = STATE["last_emotion"]

                        # Ensure "Engaged" overrides boredom if the person comes back into focus and is neutral/happy
                        # This rule is important for quick recovery from bored state
                        if STATE["last_emotion"] == "Bored" and current_emotion in ["Engaged", "Confused", "Frustrated"]:
                             # If they are no longer exhibiting bored characteristics, allow a quick switch
                            if current_emotion != "Bored":  # The previous logic was "dom_emotion in ['happy', 'neutral']" which is too narrow.
                                pass  # Let the determined current emotion stand

                        # --- State Update and Smoothing ---
                        STATE["emotion_history"].append(current_emotion)
                        smoothed_emotion = get_majority_emotion(STATE["emotion_history"])

                        with STATE["lock"]:
                            if STATE["last_emotion"] != smoothed_emotion:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                STATE["log_data"].append({"timestamp": timestamp, "emotion": smoothed_emotion, "source": "facial"})
                            STATE["last_emotion"] = smoothed_emotion

                else:  # No face detected
                    eyes_closed_start = None
                    yawn_start = None
                    gaze_away_counter = 0

                    if no_face_start is None:
                        no_face_start = time.time()
                    elif time.time() - no_face_start > NO_FACE_DURATION_BORED:
                        # After NO_FACE_DURATION_BORED, set to Bored if still no face
                        with STATE["lock"]:
                            if STATE["last_emotion"] != "Bored":
                                STATE["last_emotion"] = "Bored"
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                STATE["log_data"].append({"timestamp": timestamp, "emotion": "Bored", "source": "facial - no face (prolonged)"})
                    else:
                        # Immediately "Face Not Detected" if no face and not yet bored by duration
                        current_emotion = "Face Not Detected"
                        with STATE["lock"]:
                             # Allow immediate update for Face Not Detected
                            if STATE["last_emotion"] != "Face Not Detected":
                                STATE["last_emotion"] = "Face Not Detected"
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                STATE["log_data"].append({"timestamp": timestamp, "emotion": "Face Not Detected", "source": "facial - no face (instant)"})
            except Exception as e:
                logger.error(f"Error in facial emotion loop frame processing: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(0.1)  # Longer sleep on error to recover

    except Exception as e:
        logger.error(f"Fatal error in facial emotion detection loop: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        with STATE["lock"]:
            if STATE.get("loop_ident") == my_id:
                STATE["running"] = False
        if cap is loop_cap:
            safe_camera_release()
        # Do not close global FaceMesh here: rapid camera toggles run stop/start in parallel threads;
        # double-close raised NoneType errors, and a closed Mesh cannot run the next calibration loop.

# --- Control Functions (kept as is) ---

def start_facial_emotion_detection():
    """Start facial emotion detection in a daemon thread."""
    try:
        with STATE["lock"]:
            if not STATE["running"]:
                STATE["running"] = True
                threading.Thread(target=facial_emotion_loop, daemon=True).start()
                logger.info("Facial emotion detection started successfully.")
                logger.info(f"DEBUG - STATE['running'] set to: {STATE['running']}")
            else:
                logger.info("Facial emotion detection is already running.")
                logger.info(f"DEBUG - STATE['running'] already: {STATE['running']}")
    except Exception as e:
        logger.error(f"Failed to start facial detection: {str(e)}")
        logger.debug(traceback.format_exc())
        with STATE["lock"]:
            STATE["running"] = False

def stop_facial_emotion_detection():
    """Stop facial emotion detection and save log data."""
    try:
        with STATE["lock"]:
            if STATE["running"]:
                STATE["running"] = False
                logger.info("Attempting to stop facial emotion detection...")
        time.sleep(0.5)  # Give a moment for the loop to naturally exit
        
        safe_camera_release()

        # FaceMesh is closed in facial_emotion_loop's finally — avoid double close (NoneType errors).

        if STATE["log_data"]:
            try:
                file_exists = os.path.exists("emotion_log.csv")
                pd.DataFrame(STATE["log_data"]).to_csv("emotion_log.csv", mode='a', index=False, header=not file_exists)
                logger.info(f"Emotion log saved to emotion_log.csv. Total entries: {len(STATE['log_data'])}.")
            except Exception as e:
                logger.error(f"Error saving facial log data: {str(e)}")
                logger.debug(traceback.format_exc())
            finally:
                # Clear log_data after saving to avoid duplicate entries on subsequent runs
                STATE["log_data"].clear()
        else:
            logger.info("No emotion data was collected to save.")
        logger.info("Facial emotion detection stopped.")
    except Exception as e:
        logger.error(f"Error stopping facial detection: {str(e)}")
        logger.debug(traceback.format_exc())

def get_facial_emotion():
    """Retrieve the current smoothed facial emotion."""
    try:
        with STATE["lock"]:
            if not STATE["running"]:
                return "Unknown"
            current_emotion = STATE["last_emotion"]
            logger.debug(
                "Facial running=%s emotion=%s", STATE["running"], current_emotion
            )
            return current_emotion
    except Exception as e:
        logger.error(f"Error getting facial emotion: {str(e)}")
        logger.debug(traceback.format_exc())
        return "Unknown"

# Example of how to run and stop for testing
if __name__ == "__main__":
    set_camera_session_allowed(True)
    start_facial_emotion_detection()
    try:
        while True:
            current_face_emotion = get_facial_emotion()
            logger.info(f"Current Facial Emotion: {current_face_emotion}")
            time.sleep(1)  # Check emotion every second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Stopping facial emotion detection.")
    finally:
        stop_facial_emotion_detection()