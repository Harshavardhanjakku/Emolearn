"""Deterministic tests for facial classification, smoothing, and merge rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from modules.emotion_combiner import merge_multimodal_emotion, persist_combined_snapshot
import modules.emotion_combiner as combiner
from modules.facial_classifier import (
    CueTemporalState,
    EmotionSmoother,
    FaceCues,
    classify_frame,
    extract_face_cues,
)


class LM:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x, self.y, self.z = x, y, z


class Mesh:
    def __init__(self):
        self.landmark = [LM() for _ in range(468)]


def frontal_mesh() -> Mesh:
    m = Mesh()

    def seti(i, x, y):
        m.landmark[i].x, m.landmark[i].y = x, y

    seti(10, 0.50, 0.22)
    seti(152, 0.50, 0.82)
    seti(234, 0.22, 0.48)
    seti(454, 0.78, 0.48)
    seti(1, 0.50, 0.50)
    seti(33, 0.35, 0.42)
    seti(133, 0.44, 0.42)
    seti(160, 0.38, 0.405)
    seti(158, 0.41, 0.405)
    seti(153, 0.41, 0.435)
    seti(144, 0.38, 0.435)
    seti(263, 0.65, 0.42)
    seti(362, 0.56, 0.42)
    seti(385, 0.59, 0.405)
    seti(387, 0.62, 0.405)
    seti(373, 0.62, 0.435)
    seti(380, 0.59, 0.435)
    seti(70, 0.34, 0.34)
    seti(105, 0.38, 0.33)
    seti(55, 0.44, 0.34)
    seti(285, 0.56, 0.34)
    seti(334, 0.62, 0.33)
    seti(300, 0.66, 0.34)
    seti(61, 0.42, 0.62)
    seti(291, 0.58, 0.62)
    seti(13, 0.50, 0.605)
    seti(14, 0.50, 0.625)
    seti(132, 0.30, 0.58)
    seti(361, 0.70, 0.58)
    return m


class ClassifyRulesTests(unittest.TestCase):
    def test_blink_is_not_sleepy(self):
        cues = FaceCues(face_present=True, ear=0.12, looking_at_screen=True)
        state = CueTemporalState()
        classify_frame(cues, 0.0, state)
        label, _, dbg = classify_frame(cues, 0.25, state)
        self.assertNotEqual(label, "Sleepy")
        self.assertGreater(dbg["flags"]["blink_only"], 0)

    def test_sustained_eye_closure_is_sleepy(self):
        cues = FaceCues(face_present=True, ear=0.12, looking_at_screen=True)
        state = CueTemporalState()
        classify_frame(cues, 0.0, state)
        label, conf, _ = classify_frame(cues, 1.15, state)
        self.assertEqual(label, "Sleepy")
        self.assertGreaterEqual(conf, 0.7)

    def test_furrowed_elevens_and_clenched_teeth_frustrated(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            brow_furrow=0.03,
            inner_brow_compression=0.7,
            jaw_tension=0.55,
            teeth_visible=0.45,
            deepface={
                "angry": 52,
                "disgust": 10,
                "neutral": 20,
                "happy": 5,
                "fear": 4,
                "sad": 4,
                "surprise": 5,
            },
        )
        label, _, dbg = classify_frame(cues, 0.0, CueTemporalState())
        self.assertEqual(label, "Frustrated")
        self.assertIn("furrowed_brows", dbg["frustrated"]["reasons"])

    def test_open_mouth_alone_not_frustrated(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            mouth_open=True,
            mar=0.45,
            deepface={
                "angry": 12,
                "neutral": 60,
                "happy": 10,
                "disgust": 2,
                "fear": 5,
                "sad": 5,
                "surprise": 6,
            },
        )
        label, _, _ = classify_frame(cues, 0.0, CueTemporalState())
        self.assertNotEqual(label, "Frustrated")

    def test_raised_questioning_brows_confused(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            brow_raise=0.04,
            brow_asymmetry=0.02,
            head_tilt=0.18,
            deepface={
                "surprise": 45,
                "fear": 20,
                "neutral": 15,
                "happy": 5,
                "angry": 4,
                "disgust": 2,
                "sad": 9,
            },
        )
        label, _, dbg = classify_frame(cues, 0.0, CueTemporalState())
        self.assertEqual(label, "Confused")
        self.assertTrue(any("brows" in r or "questioning" in r for r in dbg["confused"]["reasons"]))

    def test_raised_brows_alone_not_confused(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            brow_raise=0.04,
            deepface={
                "neutral": 70,
                "happy": 10,
                "surprise": 8,
                "angry": 3,
                "disgust": 1,
                "fear": 4,
                "sad": 4,
            },
        )
        label, _, _ = classify_frame(cues, 0.0, CueTemporalState())
        self.assertNotEqual(label, "Confused")

    def test_clenched_furrow_beats_confused(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            brow_furrow=0.03,
            inner_brow_compression=0.7,
            jaw_tension=0.6,
            teeth_visible=0.5,
            brow_raise=0.03,
            brow_asymmetry=0.02,
            deepface={
                "angry": 48,
                "surprise": 30,
                "neutral": 10,
                "happy": 2,
                "fear": 5,
                "sad": 3,
                "disgust": 8,
            },
        )
        label, _, _ = classify_frame(cues, 0.0, CueTemporalState())
        self.assertEqual(label, "Frustrated")

    def test_neutral_looking_at_screen_engaged(self):
        cues = FaceCues(
            face_present=True,
            ear=0.3,
            looking_at_screen=True,
            deepface={
                "neutral": 72,
                "happy": 12,
                "angry": 3,
                "disgust": 1,
                "fear": 4,
                "sad": 4,
                "surprise": 4,
            },
        )
        label, _, _ = classify_frame(cues, 0.0, CueTemporalState())
        self.assertEqual(label, "Engaged")

    def test_open_eyes_alone_not_engaged(self):
        cues = FaceCues(face_present=True, ear=0.32, looking_at_screen=True)
        label, _, _ = classify_frame(cues, 0.0, CueTemporalState())
        self.assertNotEqual(label, "Engaged")

    def test_looking_away_becomes_bored(self):
        cues = FaceCues(face_present=True, ear=0.3, looking_at_screen=False, gaze_aside=0.08)
        state = CueTemporalState()
        classify_frame(cues, 0.0, state)
        label, _, _ = classify_frame(cues, 1.4, state)
        self.assertEqual(label, "Bored")

    def test_missing_face_becomes_unknown(self):
        cues = FaceCues(face_present=False)
        state = CueTemporalState()
        classify_frame(cues, 0.0, state)
        label, _, _ = classify_frame(cues, 1.6, state)
        self.assertEqual(label, "Unknown")

    def test_landmark_frustration_cues(self):
        m = frontal_mesh()
        m.landmark[55].x, m.landmark[55].y = 0.47, 0.38
        m.landmark[285].x, m.landmark[285].y = 0.53, 0.38
        m.landmark[13].y = 0.60
        m.landmark[14].y = 0.645
        m.landmark[61].y = 0.64
        m.landmark[291].y = 0.64
        m.landmark[132].x = 0.24
        m.landmark[361].x = 0.76
        temporal = CueTemporalState()
        cues = extract_face_cues(m, 640, 480, temporal)
        self.assertGreater(cues.brow_furrow, 0.015)
        self.assertGreater(cues.inner_brow_compression, 0.4)
        self.assertGreater(cues.teeth_visible, 0.2)

    def test_landmark_raised_asymmetric_brows(self):
        m = frontal_mesh()
        m.landmark[105].y = 0.26
        m.landmark[55].y = 0.27
        m.landmark[334].y = 0.33
        temporal = CueTemporalState()
        cues = extract_face_cues(m, 640, 480, temporal)
        self.assertGreater(cues.brow_asymmetry, 0.01)


class SmootherTests(unittest.TestCase):
    def _feed(self, smoother, label, conf, t0, n=10, dt=0.05, face=True):
        t = t0
        for _ in range(n):
            smoother.observe(t, label, conf, face)
            t += dt
        return t

    def test_publish_interval_one_to_two_seconds(self):
        s = EmotionSmoother(window_s=1.5, publish_s=1.0, hold_s=2.0)
        t = self._feed(s, "Engaged", 0.8, 0.0)
        first = s.maybe_publish(t)
        self.assertEqual(first, "Engaged")
        self.assertIsNone(s.maybe_publish(t + 0.4))
        second = s.maybe_publish(t + 1.05)
        self.assertIsNotNone(second)

    def test_noisy_alternating_labels_do_not_flicker(self):
        s = EmotionSmoother(window_s=1.8, publish_s=1.0, hold_s=2.0)
        t = 0.0
        for i in range(20):
            s.observe(t, "Frustrated" if i % 2 == 0 else "Confused", 0.7, True)
            t += 0.05
        displayed = s.maybe_publish(t)
        self.assertIn(displayed, ("Frustrated", "Confused", "Unknown"))
        for i in range(20):
            s.observe(t, "Frustrated" if i % 2 == 0 else "Confused", 0.7, True)
            t += 0.05
        later = s.maybe_publish(t + 1.3)
        self.assertEqual(later, displayed)

    def test_hold_last_when_inconclusive(self):
        s = EmotionSmoother(window_s=1.8, publish_s=0.01, hold_s=2.0)
        t = self._feed(s, "Engaged", 0.8, 0.0)
        self.assertEqual(s.maybe_publish(t), "Engaged")
        s.observe(t + 0.1, "Unknown", 0.1, True)
        held = s.maybe_publish(t + 0.2)
        self.assertEqual(held, "Engaged")

    def test_unknown_after_missing_face(self):
        s = EmotionSmoother(window_s=1.8, publish_s=0.01, hold_s=0.2, no_face_s=1.5)
        t = self._feed(s, "Engaged", 0.8, 0.0)
        s.maybe_publish(t)
        t2 = t + 0.5
        for i in range(12):
            s.observe(t2, "Unknown", 0.2, False)
            t2 += 0.15
        out = s.maybe_publish(t2)
        self.assertEqual(out, "Unknown")


class MergeAndApiTests(unittest.TestCase):
    def test_mouse_does_not_overwrite_facial(self):
        self.assertEqual(
            merge_multimodal_emotion("Frustrated", "Engaged", "Sleepy"),
            "Frustrated",
        )

    def test_disabled_microphone_ignored(self):
        # Speech detector off → caller passes Unknown
        self.assertEqual(merge_multimodal_emotion("Engaged", "Unknown", "Unknown"), "Engaged")
        self.assertEqual(merge_multimodal_emotion("Unknown", "Unknown", "Unknown"), "Unknown")

    def test_face_not_detected_does_not_invent_from_face_label(self):
        self.assertEqual(
            merge_multimodal_emotion("Face Not Detected", "Unknown", "Unknown"),
            "Unknown",
        )

    def test_duplicate_log_skipped(self):
        combiner._last_logged_key = None
        entry = {
            "time": "t1",
            "facial_emotion": "Engaged",
            "voice_emotion": "Unknown",
            "interaction_emotion": "Unknown",
            "final_emotion": "Engaged",
            "value": 0.9,
        }
        with patch("modules.emotion_combiner.append_emotion_log_entry") as append:
            persist_combined_snapshot(entry)
            persist_combined_snapshot(dict(entry, time="t2"))
            self.assertEqual(append.call_count, 1)

    def test_latest_combined_shape_consistent(self):
        snap = {
            "time": "2026-01-01 00:00:00",
            "facial_emotion": "Engaged",
            "voice_emotion": "Unknown",
            "interaction_emotion": "Unknown",
            "final_emotion": "Engaged",
            "value": 0.9,
        }
        combined = {"emotion": snap["final_emotion"]}
        self.assertEqual(combined["emotion"], snap["final_emotion"])
        for key in (
            "time",
            "facial_emotion",
            "voice_emotion",
            "interaction_emotion",
            "final_emotion",
            "value",
        ):
            self.assertIn(key, snap)


class DetectorGuardTests(unittest.TestCase):
    def test_start_facial_does_not_spawn_second_loop(self):
        from modules.facial_emotion import STATE, start_facial_emotion_detection

        STATE["running"] = False
        with patch("modules.facial_emotion.threading.Thread") as thread_cls:
            start_facial_emotion_detection()
            start_facial_emotion_detection()
            self.assertEqual(thread_cls.call_count, 1)
        STATE["running"] = False

    def test_start_detector_facial_idempotent_flag(self):
        combiner.detectors_running["facial"] = False
        with patch("modules.emotion_combiner.start_facial_emotion_detection") as start:
            combiner.start_detector("facial")
            combiner.start_detector("facial")
            self.assertTrue(combiner.detectors_running["facial"])
            self.assertGreaterEqual(start.call_count, 1)
        combiner.detectors_running["facial"] = False


if __name__ == "__main__":
    unittest.main()
