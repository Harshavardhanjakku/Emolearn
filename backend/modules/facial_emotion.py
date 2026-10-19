import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from datetime import datetime
import threading
import logging
import os
from collections import deque
from types import SimpleNamespace
from urllib.request import urlretrieve
import traceback

from modules.facial_classifier import (
    CueTemporalState,
    EmotionSmoother,
    FaceCues,
    classify_frame,
    extract_face_cues,
    normalize_deepface_probs,
)

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging to match other modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Shared State and Constants ---
STATE = {
    "running": False,
    "last_emotion": "Unknown",
    "last_emotion_time": "",
    "lock": threading.Lock(),
    "log_data": [],
    "emotion_history": deque(maxlen=24),
    "camera_available": False,
    "last_camera_check": 0,
    "reopen_camera": False,
    "loop_ident": None,
    "last_frame_time": 0.0,
}

# Emotion index mapping (keep as is)
emotion_index = {"Engaged": 0, "Confused": 1, "Frustrated": 2, "Bored": 3, "Sleepy": 4, "Unknown": 5}

# DeepFace FER classes are angry/disgust/fear/happy/sad/surprise/neutral.
# Product labels are produced in facial_classifier.py from scores + landmarks.
DEEPFACE_MIN_INTERVAL_S = 0.35
CAMERA_RETRY_INTERVAL = 1

# --- MediaPipe Setup (Tasks FaceLandmarker; mp.solutions was removed in 1.x) ---
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)


def _ensure_face_landmarker_model():
    os.makedirs(os.path.dirname(_FACE_LANDMARKER_PATH), exist_ok=True)
    if os.path.isfile(_FACE_LANDMARKER_PATH) and os.path.getsize(_FACE_LANDMARKER_PATH) > 1000:
        return _FACE_LANDMARKER_PATH
    logger.info("Downloading MediaPipe face landmarker model...")
    urlretrieve(_FACE_LANDMARKER_URL, _FACE_LANDMARKER_PATH)
    return _FACE_LANDMARKER_PATH


class _LandmarkMesh:
    """Keeps landmarks.landmark[i] access used by classifier helpers."""

    def __init__(self, points):
        self.landmark = points


class _FaceMeshAdapter:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_face_landmarker_model()),
            running_mode=RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb_frame):
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self._landmarker.detect_for_video(image, ts)
        faces = [_LandmarkMesh(pts) for pts in (result.face_landmarks or [])]
        return SimpleNamespace(multi_face_landmarks=faces or None)


face_mesh = _FaceMeshAdapter(max_num_faces=1, min_detection_confidence=0.5)
cap = None
CAMERA_INDEX = 0
_CAMERA_INDEX_LOCK = threading.Lock()

_deepface_frame_lock = threading.Lock()
_deepface_latest_frame = None
_deepface_probs = None
_deepface_worker_started = False
_deepface_last_submit = 0.0


def _deepface_worker():
    global _deepface_latest_frame, _deepface_probs
    while True:
        if not CAMERA_SESSION_ALLOWED.is_set():
            time.sleep(0.05)
            continue
        with _deepface_frame_lock:
            frame = _deepface_latest_frame
            _deepface_latest_frame = None
        if frame is None:
            time.sleep(0.02)
            continue
        try:
            from deepface import DeepFace

            deepface_result = DeepFace.analyze(
                frame, actions=["emotion"], enforce_detection=False, silent=True
            )
            if deepface_result:
                _deepface_probs = normalize_deepface_probs(deepface_result[0].get("emotion") or {})
        except Exception:
            pass


def _ensure_deepface_worker():
    global _deepface_worker_started
    if not _deepface_worker_started:
        _deepface_worker_started = True
        threading.Thread(target=_deepface_worker, daemon=True).start()


def _submit_frame_for_deepface(frame):
    global _deepface_latest_frame, _deepface_last_submit
    if frame is None or getattr(frame, "size", 0) == 0:
        return
    now = time.time()
    if now - _deepface_last_submit < DEEPFACE_MIN_INTERVAL_S:
        return
    _deepface_last_submit = now
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


def _camera_backends():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)
    return backends


def _configure_capture(capture):
    """Keep a stable 640x480 stream. Do not set BUFFERSIZE — it drops DirectShow reads."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def _warmup_capture(capture, attempts: int = 20) -> bool:
    for _ in range(attempts):
        frame = _read_frame(capture)
        if frame is not None:
            return True
        time.sleep(0.05)
    return False


def _read_frame(capture):
    """Return a valid BGR frame, or None. Never releases the camera."""
    if capture is None or not capture.isOpened():
        return None
    ok, frame = capture.read()
    if not ok or frame is None or getattr(frame, "size", 0) == 0:
        return None
    return frame


def _try_open_index(index: int):
    for backend in _camera_backends():
        test = cv2.VideoCapture(index, backend)
        if not test.isOpened():
            test.release()
            continue
        _configure_capture(test)
        if _warmup_capture(test):
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
    """Legacy helper; classification uses CueTemporalState instead."""
    try:
        return abs(landmarks.landmark[234].y - landmarks.landmark[454].y) > 0.16
    except Exception as e:
        logger.debug(f"Error checking head tilt: {str(e)}")
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

        if not CAMERA_SESSION_ALLOWED.is_set() or not STATE["running"]:
            logger.info("Camera session stopped — exiting facial loop.")
            return

        cap = open_best_camera()
        loop_cap = cap
        if cap is None or not cap.isOpened():
            raise Exception("Failed to open any webcam. Plug in an external camera and pick it in the UI.")
        consecutive_failures = 0
        temporal = CueTemporalState()
        smoother = EmotionSmoother()
        logger.info("Proctoring camera stream started — analyzing ~1s emotion windows")

        while STATE["running"]:
            if not CAMERA_SESSION_ALLOWED.is_set():
                break
            try:
                with STATE["lock"]:
                    need_reopen = STATE.get("reopen_camera")
                    if need_reopen:
                        STATE["reopen_camera"] = False
                if need_reopen:
                    if cap is loop_cap:
                        safe_camera_release()
                    cap = open_best_camera()
                    loop_cap = cap
                    consecutive_failures = 0
                    if cap is None or not cap.isOpened():
                        time.sleep(0.4)
                        continue
                frame = _read_frame(cap)
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 30:
                        logger.warning("Waiting for camera frames (device still held open)")
                    time.sleep(0.02)
                    continue

                consecutive_failures = 0
                now = time.time()
                with STATE["lock"]:
                    STATE["last_frame_time"] = now
                _submit_frame_for_deepface(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                h, w, _ = frame.shape

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    try:
                        cues = extract_face_cues(face_landmarks, w, h, temporal)
                    except Exception:
                        logger.debug("Landmark cue extraction failed", exc_info=True)
                        cues = None
                    if cues is not None:
                        cues.deepface = _deepface_probs
                        label, conf, _dbg = classify_frame(cues, now, temporal)
                        smoother.observe(now, label, conf, True)
                else:
                    missing = FaceCues(face_present=False)
                    label, conf, _dbg = classify_frame(missing, now, temporal)
                    smoother.observe(now, label, conf, False)

                published = smoother.maybe_publish(now)
                if published is not None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with STATE["lock"]:
                        changed = STATE["last_emotion"] != published
                        STATE["last_emotion"] = published
                        STATE["last_emotion_time"] = timestamp
                        STATE["emotion_history"].append(published)
                        if changed:
                            STATE["log_data"].append(
                                {"timestamp": timestamp, "emotion": published, "source": "facial"}
                            )
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
            if STATE["running"]:
                logger.debug("Facial emotion detection is already running.")
                return
            STATE["running"] = True
            STATE["last_emotion"] = "Unknown"
            STATE["last_emotion_time"] = ""
            threading.Thread(target=facial_emotion_loop, daemon=True).start()
            logger.info("Facial emotion detection started successfully.")
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
    rec = get_facial_emotion_record()
    return rec["emotion"]


def get_facial_emotion_record():
    """Emotion plus last publish timestamp (used by API snapshots)."""
    try:
        with STATE["lock"]:
            if not STATE["running"]:
                return {"emotion": "Unknown", "time": STATE.get("last_emotion_time") or ""}
            return {
                "emotion": STATE["last_emotion"],
                "time": STATE.get("last_emotion_time") or "",
            }
    except Exception as e:
        logger.error(f"Error getting facial emotion: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"emotion": "Unknown", "time": ""}

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