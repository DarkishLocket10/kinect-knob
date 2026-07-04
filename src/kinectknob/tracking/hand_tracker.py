"""MediaPipe Tasks HandLandmarker wrapper (VIDEO mode).

VIDEO mode (vs LIVE_STREAM) keeps everything synchronous on our vision thread:
detect_for_video() returns in-line, which gives the lowest end-to-end latency
for a pipeline that always processes the freshest frame and simply drops
stale ones.
"""
from __future__ import annotations

import logging

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..types import Hand

log = logging.getLogger("kk.track")


class HandTracker:
    def __init__(self, model_path: str, num_hands: int = 2):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        log.info("HandLandmarker ready (model=%s, num_hands=%d)", model_path, num_hands)

    def process(self, rgb: np.ndarray, t: float) -> list[Hand]:
        """rgb: HxWx3 uint8 RGB (contiguous). t: monotonic seconds."""
        h, w = rgb.shape[:2]
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts_ms)

        hands: list[Hand] = []
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            z = np.array([lm.z for lm in lms], dtype=np.float32)
            cat = handedness[0]
            hands.append(Hand(pts=pts, z=z, handedness=cat.category_name, score=cat.score))
        return hands

    def close(self) -> None:
        self._landmarker.close()
