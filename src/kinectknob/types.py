"""Core data types shared across the pipeline.

Everything in this module is pure-Python + numpy so the gesture engine and its
tests never need mediapipe, opencv, or a Kinect attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# MediaPipe hand landmark indices (21-point model).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Landmarks whose mean approximates the palm centre.
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


@dataclass
class Frame:
    """One captured frame. ``rgb`` is HxWx3 uint8 in RGB order (already mirrored
    if mirroring is enabled). ``depth_mm`` — when the backend provides it — is an
    HxW uint16/float32 array of millimetre distances aligned to ``rgb`` pixels
    (0 = no reading). ``ir`` is True when ``rgb`` is really a tone-mapped
    active-IR image (Kinect v2 night mode)."""

    rgb: np.ndarray
    depth_mm: Optional[np.ndarray]
    t: float          # time.monotonic() at capture
    seq: int
    ir: bool = False


@dataclass
class Hand:
    """A detected hand in pixel coordinates."""

    pts: np.ndarray            # (21, 2) float32, pixel coords in the rgb frame
    z: np.ndarray              # (21,)  float32, mediapipe relative depth (unitless)
    handedness: str            # "Left" / "Right" (as seen after mirroring)
    score: float

    @property
    def palm_center(self) -> np.ndarray:
        return self.pts[list(PALM_LANDMARKS)].mean(axis=0)

    @property
    def size(self) -> float:
        """Characteristic hand size in pixels (wrist to middle-finger knuckle)."""
        return float(np.linalg.norm(self.pts[MIDDLE_MCP] - self.pts[WRIST]))


# A callable mapping an (x, y) pixel in the rgb frame to metres, or None if the
# backend has no depth / no reading there.
DepthSampler = Callable[[float, float], Optional[float]]


# ---------------------------------------------------------------------------
# Gesture events emitted by the engine, consumed by the controller.
# ---------------------------------------------------------------------------

@dataclass
class GestureEvent:
    t: float


@dataclass
class KnobEngage(GestureEvent):
    """User pinched: the invisible knob is now gripped."""


@dataclass
class KnobTurn(GestureEvent):
    """Knob rotated. ``deg`` is the filtered total rotation since engage
    (positive = clockwise as the user sees it = volume up unless inverted);
    ``delta_deg`` is the change since the previous KnobTurn."""

    deg: float = 0.0
    delta_deg: float = 0.0


@dataclass
class KnobRelease(GestureEvent):
    """Pinch released (or hand lost). ``deg`` is the final total rotation."""

    deg: float = 0.0


@dataclass
class Swipe(GestureEvent):
    """Open-palm horizontal swipe. ``direction`` +1 = user's right = next track,
    -1 = user's left = previous track."""

    direction: int = 1
    speed: float = 0.0     # frame-widths per second


@dataclass
class FistHold(GestureEvent):
    """Closed fist held still: play/pause toggle (optional, off by default)."""


@dataclass
class EngineSnapshot:
    """Live state for the web UI / debug overlay."""

    state: str = "idle"                 # idle | engaging | engaged
    hand_present: bool = False
    handedness: str = ""
    pinch_ratio: float = 0.0
    openness: str = ""                  # open | fist | neutral
    angle_deg: float = 0.0              # filtered total rotation while engaged
    palm_xy: tuple[float, float] = (0.0, 0.0)
    palm_speed: float = 0.0             # frame-widths / s
    hand_depth_m: Optional[float] = None
    gated_out: str = ""                 # reason a hand was ignored, for tuning
    last_event: str = ""
    extra: dict = field(default_factory=dict)
