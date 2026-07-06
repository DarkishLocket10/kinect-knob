"""Synthetic hand builders: generate realistic 21-point landmark sets so the
gesture engine is tested against the same geometry MediaPipe produces."""
from __future__ import annotations

import numpy as np
import pytest

from kinectknob.config import AppConfig
from kinectknob.types import Hand

FRAME_W, FRAME_H = 640, 480
FPS = 30.0

# Canonical open hand, wrist at origin, fingers pointing up (image coords,
# y down). Units: pixels, hand size (wrist -> middle MCP) = 100.
_BASE = np.array([
    (0, 0),                                          # 0 wrist
    (-30, -20), (-45, -45), (-55, -65), (-65, -85),  # thumb
    (-35, -95), (-38, -130), (-40, -155), (-42, -180),   # index
    (0, -100), (0, -140), (0, -165), (0, -195),          # middle
    (30, -95), (32, -132), (34, -155), (36, -178),       # ring
    (55, -85), (57, -110), (59, -128), (61, -145),       # pinky
], dtype=np.float64)

_FINGER_TIPS = (8, 12, 16, 20)
_FINGER_PIPS = (6, 10, 14, 18)


def make_hand(
    center=(320.0, 300.0),
    angle_deg: float = 0.0,
    pose: str = "open",          # open | pinch | fist | release
    scale: float = 1.0,
    score: float = 0.95,
) -> Hand:
    pts = _BASE.copy() * scale

    if pose == "pinch":
        # Thumb tip meets index tip (like gripping a small knob).
        grip = np.array([-40.0, -100.0]) * scale
        pts[4] = grip + (scale * 6, 0)
        pts[8] = grip - (scale * 6, 0)
    elif pose == "fist":
        # Fold all four fingers: tips pulled back near the knuckles.
        for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS):
            pts[tip] = pts[pip] * 0.55
        pts[4] = np.array([-35.0, -60.0]) * scale
    elif pose == "release":
        # Thumb and index clearly apart (past the release hysteresis).
        pts[4] = np.array([-90.0, -60.0]) * scale

    # Rotate about the palm centre. In image coords (y down) a positive angle
    # here is clockwise on screen — matching the engine's sign convention.
    th = np.radians(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    palm_ids = [0, 5, 9, 13, 17]
    palm_center = pts[palm_ids].mean(axis=0)
    pts = (pts - palm_center) @ rot.T + palm_center

    pts = pts + np.asarray(center, dtype=np.float64) - palm_center
    return Hand(
        pts=pts.astype(np.float32),
        z=np.zeros(21, dtype=np.float32),
        handedness="Right",
        score=score,
    )


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


class Timeline:
    """Feeds hands into an engine at a fixed frame rate, collecting events."""

    def __init__(self, engine):
        self.engine = engine
        self.t = 0.0
        self.events = []

    def step(self, hands, n: int = 1, depth_sampler=None):
        out = []
        for _ in range(n):
            self.t += 1.0 / FPS
            evs = self.engine.update(hands, self.t, FRAME_W, FRAME_H, depth_sampler)
            out.extend(evs)
        self.events.extend(out)
        return out
