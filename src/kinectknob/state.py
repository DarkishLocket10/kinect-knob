"""Thread-safe shared state between the vision thread and the web server."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from .types import EngineSnapshot, Frame, Hand


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: Optional[np.ndarray] = None          # proc-sized RGB frame
        self._hands: list[Hand] = []
        self._engine: EngineSnapshot = EngineSnapshot()
        self.fps: float = 0.0
        self.proc_ms: float = 0.0
        self.backend: str = ""
        self.has_depth: bool = False
        self.ir_active: bool = False
        self.started_at: float = time.time()
        self.last_frame_t: float = 0.0                  # monotonic
        self._fullres: Optional[np.ndarray] = None      # unmirrored BGR 1080p
        self._fullres_t: float = 0.0
        # Backend hook for stacked "proper photos" (capture_photo); set once
        # by main before the threads start, called from the web executor.
        self.photo_fn: Optional[Callable[[int, float], Optional[np.ndarray]]] = None

    def update_vision(
        self,
        rgb: np.ndarray,
        hands: list[Hand],
        engine: EngineSnapshot,
        fps: float,
        proc_ms: float,
        ir: bool = False,
    ) -> None:
        with self._lock:
            self._rgb = rgb
            self._hands = hands
            self._engine = engine
            self.fps = fps
            self.proc_ms = proc_ms
            self.ir_active = ir
            self.last_frame_t = time.monotonic()

    def update_fullres(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._fullres = bgr
            self._fullres_t = time.monotonic()

    def fullres(self) -> tuple[Optional[np.ndarray], float]:
        """Latest full-resolution UNMIRRORED BGR frame and its capture time."""
        with self._lock:
            return self._fullres, self._fullres_t

    def render_data(self) -> tuple[Optional[np.ndarray], list[Hand], EngineSnapshot]:
        with self._lock:
            return self._rgb, list(self._hands), self._engine

    def healthy(self, max_age_s: float = 5.0) -> bool:
        with self._lock:
            return self.last_frame_t > 0 and (time.monotonic() - self.last_frame_t) < max_age_s

    def state_dict(self) -> dict:
        with self._lock:
            eng = self._engine
            return {
                "backend": self.backend,
                "has_depth": self.has_depth,
                "ir_active": self.ir_active,
                "fps": round(self.fps, 1),
                "proc_ms": round(self.proc_ms, 1),
                "uptime_s": int(time.time() - self.started_at),
                "engine": {
                    "state": eng.state,
                    "hand_present": eng.hand_present,
                    "handedness": eng.handedness,
                    "pinch_ratio": eng.pinch_ratio,
                    "curl_gap": eng.curl_gap,
                    "openness": eng.openness,
                    "angle_deg": eng.angle_deg,
                    "palm_speed": eng.palm_speed,
                    "hand_depth_m": eng.hand_depth_m,
                    "gated_out": eng.gated_out,
                    "last_event": eng.last_event,
                    # e.g. {"facing": 0.78} while an open palm is up — the
                    # live value to tune playpause.facing_min against
                    "extra": dict(eng.extra),
                },
            }
