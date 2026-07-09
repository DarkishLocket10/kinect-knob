"""Low-light visibility boost: auto-gamma on dim frames before hand tracking.

Motion blur comes from long exposure. Capping the shutter (KK_EXPOSURE=semi:N)
makes fast hands sharp but the frames DARK — and MediaPipe stops finding dim
hands well before the scene is dark enough for IR night mode (mean luma < 28).
This booster closes that gap: an auto-gamma LUT lifts a dim frame's midtones
toward a target luma. Gamma (unlike plain gain) brightens shadows and midtones
without clipping the highlights the landmark model keys on.

Identity for bright scenes, EMA-smoothed so TV flicker can't strobe the
brightness, and skipped entirely for IR frames (already tone-mapped and
self-illuminated). Pure numpy — unit-testable without cv2 or a camera.
"""
from __future__ import annotations

import numpy as np

TRIGGER_LUMA = 90.0    # frames brighter than this pass through untouched
TARGET_LUMA = 110.0    # dim frames are lifted toward this mean
MIN_EXPONENT = 0.45    # brightening cap (≈ gamma 2.2)
_EMA_ALPHA = 0.15      # per-frame smoothing toward the desired exponent


class LowLightBoost:
    def __init__(self) -> None:
        self._exponent = 1.0
        self._lut_exponent = 1.0
        self._lut: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self._exponent < 0.98

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: HxWx3 uint8. Returns the frame, brightened when dim."""
        mean = float(rgb[::8, ::8].mean())
        if mean < 1.0:                       # essentially black: cap the lift
            desired = MIN_EXPONENT
        elif mean >= TRIGGER_LUMA:
            desired = 1.0
        else:
            # (mean/255)^e == TARGET/255  =>  e = ln(TARGET/255)/ln(mean/255)
            desired = np.log(TARGET_LUMA / 255.0) / np.log(mean / 255.0)
            desired = float(min(max(desired, MIN_EXPONENT), 1.0))
        self._exponent += _EMA_ALPHA * (desired - self._exponent)
        if not self.active:
            return rgb
        if self._lut is None or abs(self._exponent - self._lut_exponent) > 0.01:
            self._lut = np.rint(
                ((np.arange(256) / 255.0) ** self._exponent) * 255.0
            ).astype(np.uint8)
            self._lut_exponent = self._exponent
        return self._lut[rgb]
