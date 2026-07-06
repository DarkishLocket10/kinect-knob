"""Kinect v2 active-IR support: night-mode auto-switch + IR tone mapping.

The v2's time-of-flight sensor carries its own IR illuminator, so the IR
stream is fully exposed in a pitch-black room — the one thing the color
camera can't do. When the room gets too dark for RGB hand tracking we feed
MediaPipe the tone-mapped IR image instead. Bonus: IR and depth come off the
same sensor, so in IR mode the depth map is pixel-aligned for free (no
registration pass needed).

Pure numpy — no cv2, no freenect2 — so the switching logic and tone map are
unit-testable anywhere.
"""
from __future__ import annotations

import numpy as np

# Mean color-frame luma (0-255) thresholds with hysteresis: below DARK_LUMA we
# consider the room too dark for RGB tracking, above BRIGHT_LUMA it's bright
# enough to switch back. The gap prevents flapping at dusk / TV flicker.
DARK_LUMA = 28.0
BRIGHT_LUMA = 48.0
# Frames the luma must stay past a threshold before switching (~0.7 s at 30fps).
DWELL_FRAMES = 20

_IR_MAX = 65535.0


class IrAutoSwitch:
    """Decides per-frame whether to track on IR instead of color.

    mode: "off" (never), "always" (unconditionally), or "auto" (switch on
    sustained darkness, back on sustained brightness).
    """

    def __init__(self, mode: str = "auto"):
        mode = mode.lower()
        if mode not in ("auto", "off", "always"):
            raise ValueError(f"ir_mode must be auto|off|always, got {mode!r}")
        self.mode = mode
        self.active = mode == "always"
        self._count = 0

    def update(self, color_mean_luma: float) -> bool:
        """Feed one frame's mean color luma (0-255); returns True to use IR."""
        if self.mode == "off":
            return False
        if self.mode == "always":
            return True
        if self.active:
            crossing = color_mean_luma > BRIGHT_LUMA
        else:
            crossing = color_mean_luma < DARK_LUMA
        self._count = self._count + 1 if crossing else 0
        if self._count >= DWELL_FRAMES:
            self.active = not self.active
            self._count = 0
        return self.active


def ir_to_rgb(ir: np.ndarray) -> np.ndarray:
    """Tone-map a raw Kinect v2 IR frame (float32, 0-65535) to HxWx3 uint8.

    Square root brings up the dim mid-range (hands at 1-3 m) without blowing
    out retro-reflective hotspots — the standard display transform for
    Kinect IR.
    """
    ir = np.asarray(ir, dtype=np.float32)
    if ir.ndim == 3 and ir.shape[-1] == 1:   # some bindings emit (H, W, 1)
        ir = ir[..., 0]
    ir8 = (np.sqrt(np.clip(ir, 0.0, _IR_MAX) / _IR_MAX) * 255.0).astype(np.uint8)
    return np.repeat(ir8[:, :, np.newaxis], 3, axis=2)
