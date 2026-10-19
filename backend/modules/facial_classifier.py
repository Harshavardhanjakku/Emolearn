"""
Facial affect classification for EmoLearn.

DeepFace (FER-style) only outputs angry/disgust/fear/happy/sad/surprise/neutral.
Labels used by the product (Confused, Sleepy, Engaged, Frustrated, Bored) are
derived by combining those scores with MediaPipe landmark cues and temporal
evidence. This module does not invent new trained classes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# DeepFace emotion keys (0-100 percentages).
DEEPFACE_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

VALID_OUTPUTS = (
    "Frustrated",
    "Confused",
    "Sleepy",
    "Bored",
    "Engaged",
    "Unknown",
)

# Temporal / smoothing
PUBLISH_INTERVAL_S = 1.0
WINDOW_S = 1.5
HOLD_LAST_S = 2.0
NO_FACE_UNKNOWN_S = 1.5
MIN_VOTE_RATIO = 0.48
MIN_VOTE_COUNT = 4
HYSTERESIS_RATIO = 0.12
MIN_CONFIDENCE = 0.38

# Sleepy vs blink (typical blink ~100-400ms)
EYES_CLOSED_SLEEPY_S = 1.0
EAR_SLEEPY = 0.18
EAR_DROOP = 0.22

# Boredom: gaze fully outside the screen for a sustained period
GAZE_AWAY_BORED_S = 1.8
GAZE_ASIDE_THRESHOLD = 0.055
GAZE_PITCH_THRESHOLD = 0.035
GAZE_OFF_SCREEN_ASIDE = 0.09
GAZE_OFF_SCREEN_DOWN = 0.08
GAZE_OFF_SCREEN_UP = -0.10
LOOK_UP_THRESHOLD = -0.038
SUDDEN_TILT_MIN = 0.10
SUDDEN_TILT_DELTA = 0.035

# Yawn supports Sleepy, not Bored
MAR_YAWN = 0.55
YAWN_SLEEPY_S = 0.7

# Frustration landmark proxies (not literal wrinkle detection)
FURROW_THRESHOLD = 0.018
INNER_BROW_COMPRESS = 0.55
JAW_TENSION_THRESHOLD = 0.42
TEETH_VISIBLE_MIN = 0.28
TEETH_VISIBLE_MAX = 0.72
MAR_TALKING = 0.38  # open mouth from speech — not frustration by itself

# Confusion
BROW_RAISE_THRESHOLD = 0.028
BROW_ASYM_THRESHOLD = 0.012
HEAD_TILT_CONFUSED = 0.16  # roll; used only with other cues

# DeepFace % thresholds
ANGRY_FRUST = 40.0
ANGRY_EXTREME = 58.0
CALM_MARGIN = 8.0
SURPRISE_CONFUSED = 32.0
FEAR_SAD_CONFUSED = 38.0
NEUTRAL_ENGAGED = 34.0
HAPPY_ENGAGED = 30.0


@dataclass
class FaceCues:
    face_present: bool
    ear: float = 0.0
    mar: float = 0.0
    brow_furrow: float = 0.0
    inner_brow_compression: float = 0.0
    brow_raise: float = 0.0
    brow_asymmetry: float = 0.0
    jaw_tension: float = 0.0
    teeth_visible: float = 0.0
    mouth_open: bool = False
    smiling: bool = False
    gaze_aside: float = 0.0
    gaze_pitch: float = 0.0
    head_tilt: float = 0.0
    looking_at_screen: bool = True
    hand_near_face: bool = False
    deepface: Optional[Dict[str, float]] = None


@dataclass
class CueTemporalState:
    eyes_closed_start: Optional[float] = None
    yawn_start: Optional[float] = None
    gaze_away_start: Optional[float] = None
    no_face_start: Optional[float] = None
    pose_pitch: Deque = field(default_factory=lambda: deque(maxlen=20))
    pose_yaw: Deque = field(default_factory=lambda: deque(maxlen=20))
    baseline_pitch: Optional[float] = None
    baseline_yaw: Optional[float] = None
    last_head_tilt: float = 0.0

    def reset_face_timers(self) -> None:
        self.eyes_closed_start = None
        self.yawn_start = None
        self.gaze_away_start = None


def _df(cues: FaceCues, key: str) -> float:
    if not cues.deepface:
        return 0.0
    return float(cues.deepface.get(key, 0.0) or 0.0)


def _dominant_deepface(cues: FaceCues) -> str:
    if not cues.deepface:
        return ""
    return max(DEEPFACE_CLASSES, key=lambda k: _df(cues, k))


def extract_face_cues(landmarks, w: int, h: int, temporal: CueTemporalState) -> FaceCues:
    """Compute geometry cues from a MediaPipe-like landmark list (normalized x,y,z)."""
    def pt(i: int):
        lm = landmarks.landmark[i]
        return float(lm.x), float(lm.y)

    def dist(i: int, j: int) -> float:
        a, b = pt(i), pt(j)
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def ear(ids) -> float:
        p = [pt(i) for i in ids]
        a = ((p[1][0] - p[5][0]) ** 2 + (p[1][1] - p[5][1]) ** 2) ** 0.5
        b = ((p[2][0] - p[4][0]) ** 2 + (p[2][1] - p[4][1]) ** 2) ** 0.5
        c = ((p[0][0] - p[3][0]) ** 2 + (p[0][1] - p[3][1]) ** 2) ** 0.5
        return (a + b) / (2.0 * c) if c else 0.0

    face_w = max(dist(234, 454), 1e-6)
    face_h = max(dist(10, 152), 1e-6)

    left_ear = ear([33, 160, 158, 133, 153, 144])
    right_ear = ear([362, 385, 387, 263, 373, 380])
    ear_v = (left_ear + right_ear) / 2.0

    # MAR: vertical lip opening / mouth width
    mar = dist(13, 14) / max(dist(61, 291), 1e-6)

    inner_l, inner_r = pt(55), pt(285)
    outer_l, outer_r = pt(70), pt(300)
    mid_l, mid_r = pt(105), pt(334)
    eye_l, eye_r = pt(33), pt(263)
    eye_mid_y = (eye_l[1] + eye_r[1]) / 2.0
    brow_inner_y = (inner_l[1] + inner_r[1]) / 2.0
    brow_outer_y = (outer_l[1] + outer_r[1]) / 2.0
    brow_mid_y = (mid_l[1] + mid_r[1]) / 2.0

    # Inner brows lower than outer → furrow; closer together → "11" proxy
    furrow = ((inner_l[1] - outer_l[1]) + (inner_r[1] - outer_r[1])) / 2.0
    inner_span = dist(55, 285) / face_w
    compression = max(0.0, min(1.0, (0.22 - inner_span) / 0.12))

    raise_amt = eye_mid_y - brow_mid_y
    left_raise = eye_l[1] - mid_l[1]
    right_raise = eye_r[1] - mid_r[1]
    brow_asym = abs(left_raise - right_raise)

    lip_upper, lip_lower = pt(13), pt(14)
    corner_l, corner_r = pt(61), pt(291)
    lip_center_y = (lip_upper[1] + lip_lower[1]) / 2.0
    smiling = (corner_l[1] + corner_r[1]) / 2.0 < lip_center_y - 0.008
    lip_gap = dist(13, 14) / face_h
    mouth_open = mar > 0.22

    # Clenched / teeth-baring: parted lips, not a smile, not a wide yawn
    teeth = 0.0
    if 0.018 <= lip_gap <= 0.07 and not smiling and mar < MAR_YAWN:
        teeth = min(1.0, lip_gap / 0.05)

    jaw_w = dist(132, 361) / face_w
    # Tension: relatively wide jaw with compressed or gritted mouth
    jaw_tension = 0.0
    if jaw_w > 0.62 and mar < 0.35 and not smiling:
        jaw_tension = min(1.0, (jaw_w - 0.58) / 0.2 + (0.35 - mar))

    nose = pt(1)
    mid_x = (pt(234)[0] + pt(454)[0]) / 2.0
    yaw = nose[0] - mid_x
    pitch = (nose[1] - eye_mid_y) / face_h
    tilt = abs(pt(234)[1] - pt(454)[1])

    if temporal.baseline_pitch is None and len(temporal.pose_pitch) >= 8:
        temporal.baseline_pitch = sum(temporal.pose_pitch) / len(temporal.pose_pitch)
        temporal.baseline_yaw = sum(temporal.pose_yaw) / len(temporal.pose_yaw)
    temporal.pose_pitch.append(pitch)
    temporal.pose_yaw.append(yaw)

    bp = temporal.baseline_pitch if temporal.baseline_pitch is not None else pitch
    by = temporal.baseline_yaw if temporal.baseline_yaw is not None else yaw
    gaze_pitch = pitch - bp
    gaze_aside = abs(yaw - by)

    looking_at_screen = gaze_aside < GAZE_ASIDE_THRESHOLD

    return FaceCues(
        face_present=True,
        ear=ear_v,
        mar=mar,
        brow_furrow=furrow,
        inner_brow_compression=compression,
        brow_raise=raise_amt,
        brow_asymmetry=brow_asym,
        jaw_tension=jaw_tension,
        teeth_visible=teeth,
        mouth_open=mouth_open,
        smiling=smiling,
        gaze_aside=gaze_aside,
        gaze_pitch=gaze_pitch,
        head_tilt=tilt,
        looking_at_screen=looking_at_screen,
    )


def update_temporal(cues: FaceCues, now: float, state: CueTemporalState) -> Dict[str, float]:
    """Track durations. Returns flags used by the frame classifier."""
    flags = {
        "sleepy_eyes": 0.0,
        "yawn": 0.0,
        "gaze_away": 0.0,
        "no_face": 0.0,
        "blink_only": 0.0,
        "sudden_tilt": 0.0,
    }
    if not cues.face_present:
        state.reset_face_timers()
        if state.no_face_start is None:
            state.no_face_start = now
        flags["no_face"] = now - state.no_face_start
        return flags

    state.no_face_start = None

    if cues.ear < EAR_SLEEPY:
        if state.eyes_closed_start is None:
            state.eyes_closed_start = now
        closed_for = now - state.eyes_closed_start
        if closed_for >= EYES_CLOSED_SLEEPY_S:
            flags["sleepy_eyes"] = closed_for
        else:
            flags["blink_only"] = closed_for
    else:
        state.eyes_closed_start = None

    if cues.mar >= MAR_YAWN:
        if state.yawn_start is None:
            state.yawn_start = now
        yawn_for = now - state.yawn_start
        if yawn_for >= YAWN_SLEEPY_S:
            flags["yawn"] = yawn_for
    else:
        state.yawn_start = None

    tilt_delta = cues.head_tilt - state.last_head_tilt
    if cues.head_tilt >= SUDDEN_TILT_MIN and tilt_delta >= SUDDEN_TILT_DELTA:
        flags["sudden_tilt"] = tilt_delta
    state.last_head_tilt = cues.head_tilt

    completely_away = (
        cues.gaze_aside >= GAZE_OFF_SCREEN_ASIDE
        or cues.gaze_pitch >= GAZE_OFF_SCREEN_DOWN
        or cues.gaze_pitch <= GAZE_OFF_SCREEN_UP
    )
    off_screen = completely_away and cues.ear >= EAR_SLEEPY
    if off_screen:
        if state.gaze_away_start is None:
            state.gaze_away_start = now
        away_for = now - state.gaze_away_start
        if away_for >= GAZE_AWAY_BORED_S:
            flags["gaze_away"] = away_for
    else:
        state.gaze_away_start = None

    return flags


def _frustration_evidence(cues: FaceCues) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    angry = max(_df(cues, "angry"), _df(cues, "disgust"))
    calm = max(_df(cues, "neutral"), _df(cues, "happy"))

    furrowed = cues.brow_furrow >= FURROW_THRESHOLD
    elevens = cues.inner_brow_compression >= INNER_BROW_COMPRESS
    tense_mouth = cues.jaw_tension >= JAW_TENSION_THRESHOLD or (
        TEETH_VISIBLE_MIN <= cues.teeth_visible <= TEETH_VISIBLE_MAX
    )
    model_angry = angry >= ANGRY_FRUST and angry >= calm + CALM_MARGIN
    model_extreme = angry >= ANGRY_EXTREME and angry >= calm + 5.0

    if furrowed:
        score += 0.28
        reasons.append("furrowed_brows")
    if elevens:
        score += 0.22
        reasons.append("inner_brow_compression")
    if tense_mouth:
        score += 0.28
        reasons.append("tense_jaw_or_teeth")
    if model_angry:
        score += 0.22
        reasons.append("deepface_angry")
    if model_extreme:
        score += 0.12
        reasons.append("deepface_angry_strong")

    # Mouth open or brow motion alone is not enough.
    if cues.mouth_open and cues.mar >= MAR_TALKING and not tense_mouth and not furrowed:
        score *= 0.35
        reasons.append("downgraded_open_mouth_only")
    if furrowed and not (elevens or tense_mouth or model_angry):
        score *= 0.55
        reasons.append("furrow_without_support")

    return min(score, 1.0), reasons


def _confusion_evidence(cues: FaceCues, flags: Dict[str, float]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    looking_up = cues.gaze_pitch <= LOOK_UP_THRESHOLD
    sudden_tilt = flags.get("sudden_tilt", 0.0) > 0.0
    furrowed = cues.brow_furrow >= FURROW_THRESHOLD

    if cues.looking_at_screen and looking_up:
        score = 0.72
        reasons.append("looking_at_screen_upwards")
    elif cues.looking_at_screen and sudden_tilt and furrowed and cues.hand_near_face:
        score = 0.72
        reasons.append("tilt_furrow_hand_near_face")

    return min(score, 1.0), reasons


def _engaged_evidence(cues: FaceCues) -> Tuple[float, List[str]]:
    eyes_open = cues.ear >= EAR_DROOP
    if not eyes_open:
        return 0.0, ["eyes_not_open"]
    if not cues.looking_at_screen:
        return 0.0, ["not_looking_at_screen"]
    return 0.9, ["gaze_on_screen"]


def classify_frame(cues: FaceCues, now: float, temporal: CueTemporalState) -> Tuple[str, float, Dict]:
    """
    One-frame hypothesis. Sleepy/Bored need duration flags; others need multi-cue scores.
    Returns (label, confidence, debug).
    """
    flags = update_temporal(cues, now, temporal)
    debug = {"flags": flags}

    if not cues.face_present:
        label = "Unknown" if flags["no_face"] >= NO_FACE_UNKNOWN_S else "_hold"
        return label, 0.2 if label == "Unknown" else 0.0, debug

    frust_s, frust_r = _frustration_evidence(cues)
    conf_s, conf_r = _confusion_evidence(cues, flags)
    eng_s, eng_r = _engaged_evidence(cues)
    debug["frustrated"] = {"score": frust_s, "reasons": frust_r}
    debug["confused"] = {"score": conf_s, "reasons": conf_r}
    debug["engaged"] = {"score": eng_s, "reasons": eng_r}

    # Priority: sustained physiology first, then conflicting affect.
    if flags["sleepy_eyes"] or (flags["yawn"] and cues.ear < EAR_DROOP):
        return "Sleepy", 0.82, debug

    if flags["gaze_away"]:
        return "Bored", 0.72, debug

    # Furrow + 11s + clenched teeth strongly prefer Frustrated over Confused.
    clenched_furrow = (
        cues.brow_furrow >= FURROW_THRESHOLD
        and cues.inner_brow_compression >= INNER_BROW_COMPRESS * 0.85
        and (cues.jaw_tension >= JAW_TENSION_THRESHOLD or cues.teeth_visible >= TEETH_VISIBLE_MIN)
    )
    if clenched_furrow and frust_s >= 0.5:
        return "Frustrated", max(frust_s, 0.7), debug

    if frust_s >= 0.55 and frust_s >= conf_s + 0.08:
        return "Frustrated", frust_s, debug

    if conf_s >= 0.5 and conf_s > frust_s:
        return "Confused", conf_s, debug

    if eng_s >= 0.55 and eng_s >= max(frust_s, conf_s):
        return "Engaged", eng_s, debug

    best = max(
        ("Frustrated", frust_s),
        ("Confused", conf_s),
        ("Engaged", eng_s),
        key=lambda x: x[1],
    )
    if best[1] >= MIN_CONFIDENCE:
        return best[0], best[1], debug

    return "Unknown", best[1], debug


@dataclass
class _Sample:
    t: float
    label: str
    confidence: float
    face_present: bool


class EmotionSmoother:
    """Rolling-window vote with 1–2s publish interval and hold-last on weak evidence."""

    def __init__(
        self,
        window_s: float = WINDOW_S,
        publish_s: float = PUBLISH_INTERVAL_S,
        hold_s: float = HOLD_LAST_S,
        no_face_s: float = NO_FACE_UNKNOWN_S,
    ):
        self.window_s = window_s
        self.publish_s = publish_s
        self.hold_s = hold_s
        self.no_face_s = no_face_s
        self.samples: Deque[_Sample] = deque()
        self.displayed = "Unknown"
        self.displayed_at = 0.0
        self.last_reliable: Optional[str] = None
        self.last_reliable_at = 0.0
        self.last_publish_at = -1e9
        self.last_change_at = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def observe(self, now: float, label: str, confidence: float, face_present: bool) -> None:
        if label == "_hold":
            # Inconclusive missing-face: do not vote Unknown yet.
            self._prune(now)
            return
        self.samples.append(_Sample(now, label, confidence, face_present))
        self._prune(now)

    def maybe_publish(self, now: float) -> Optional[str]:
        if now - self.last_publish_at < self.publish_s:
            return None
        self._prune(now)
        self.last_publish_at = now
        decided = self._decide(now)
        if decided != self.displayed:
            self.displayed = decided
            self.last_change_at = now
        self.displayed_at = now
        return self.displayed

    def _decide(self, now: float) -> str:
        if not self.samples:
            if self.last_reliable and now - self.last_reliable_at <= self.hold_s:
                return self.last_reliable
            return "Unknown"

        no_face_n = sum(1 for s in self.samples if not s.face_present or s.label == "Unknown")
        if no_face_n == len(self.samples):
            oldest = self.samples[0].t
            if now - oldest >= self.no_face_s or all(not s.face_present for s in self.samples):
                if self.last_reliable and now - self.last_reliable_at <= self.hold_s and no_face_n < 3:
                    return self.last_reliable
                return "Unknown"

        votes: Dict[str, List[_Sample]] = {}
        for s in self.samples:
            if s.label in ("Unknown", "_hold") or not s.face_present:
                continue
            if s.confidence < MIN_CONFIDENCE and s.label != self.displayed:
                continue
            votes.setdefault(s.label, []).append(s)

        if not votes:
            if self.last_reliable and now - self.last_reliable_at <= self.hold_s:
                return self.last_reliable
            return "Unknown"

        ranked = sorted(
            votes.items(),
            key=lambda kv: (
                kv[0] == self.displayed,
                len(kv[1]),
                sum(x.confidence for x in kv[1]) / len(kv[1]),
            ),
            reverse=True,
        )
        winner, w_samples = ranked[0]
        n = len(w_samples)
        total_valid = sum(len(v) for v in votes.values())
        ratio = n / max(total_valid, 1)
        avg_c = sum(x.confidence for x in w_samples) / n

        enough = n >= MIN_VOTE_COUNT and ratio >= MIN_VOTE_RATIO and avg_c >= MIN_CONFIDENCE
        if self.displayed not in ("Unknown", winner) and ranked:
            # Need extra evidence to switch away from a stable label.
            current_n = len(votes.get(self.displayed, []))
            if n < current_n + max(2, int(total_valid * HYSTERESIS_RATIO)) and not enough:
                return self.displayed

        if enough:
            self.last_reliable = winner
            self.last_reliable_at = now
            return winner

        if self.last_reliable and now - self.last_reliable_at <= self.hold_s:
            return self.last_reliable
        return "Unknown"


def normalize_deepface_probs(emotion_probabilities: Dict) -> Dict[str, float]:
    """Accept 0-1 or 0-100 DeepFace dumps and return 0-100."""
    out = {k: float(emotion_probabilities.get(k, 0) or 0) for k in DEEPFACE_CLASSES}
    peak = max(out.values()) if out else 0.0
    if peak <= 1.5:
        out = {k: v * 100.0 for k, v in out.items()}
    return out
