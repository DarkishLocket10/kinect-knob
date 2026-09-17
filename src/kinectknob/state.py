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
        # Backend hooks, set once by main before the threads start and called
        # from the web executor: stacked "proper photos" (capture_photo),
        # stacked active-IR photos (capture_ir_photo), and aligned-depth
        # region stats (region_depth) for the whiteboard obstruction check.
        self.photo_fn: Optional[Callable[[int, float], Optional[np.ndarray]]] = None
        self.ir_photo_fn: Optional[Callable[[int, float], Optional[np.ndarray]]] = None
        self.region_depth_fn: Optional[
            Callable[[int, int, int, int], Optional[dict]]] = None
        # Region OCCUPANCY against a learned board/wall plane — the sensitive
        # per-region presence signal whiteboard-sync gates its scans on.
        self.region_occupancy_fn: Optional[
            Callable[[int, int, int, int], Optional[dict]]] = None
        self.relearn_fn: Optional[Callable[[], None]] = None
        # Camera lifecycle, published for the dashboard: "ok" | "absent" |
        # "error" | "down". "absent" is a WAIT, not a failure — see main.
        self.camera: str = "starting"
        self.camera_detail: str = ""
        self.camera_waiting_s: float = 0.0
        self.generation: int = 0        # capture generations opened this process
        self.recycles: int = 0          # in-process recoveries in the last window
        self.idle: bool = False         # presence says the room is empty
        self.presence = None            # presence.DepthPresence, set by App

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
            self.idle = False
            self.last_frame_t = time.monotonic()

    def update_idle(self, engine: EngineSnapshot) -> None:
        """Heartbeat for a frame the vision loop deliberately did NOT process.

        It still counts as a live frame: /healthz asks "is the camera feeding
        us", and an idling pipeline is healthy. Without this an empty room
        would flap the container healthcheck."""
        with self._lock:
            self._hands = []
            self._engine = engine
            # Zeroed, not left stale: these measure the TRACKING loop, and it
            # is deliberately not running. `idle` is what says why.
            self.fps = 0.0
            self.proc_ms = 0.0
            self.idle = True
            self.last_frame_t = time.monotonic()

    def set_camera(self, status: str, detail: str = "", waiting_s: float = 0.0) -> None:
        with self._lock:
            self.camera = status
            self.camera_detail = detail
            self.camera_waiting_s = round(waiting_s, 1)

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

    def presence_dict(self) -> dict:
        """Presence plus the camera context a caller needs to judge it: an
        absent or just-restarted camera has no opinion worth acting on, and a
        consumer that gates real behaviour on presence must be able to tell
        "nobody is here" from "I cannot see"."""
        with self._lock:
            snap = self.presence.snapshot() if self.presence is not None else None
            out = {"camera": self.camera, "idle": self.idle,
                   "has_depth": self.has_depth, "backend": self.backend,
                   "generation": self.generation}
        if snap is None:
            out["available"] = False
            return out
        out.update(snap)
        out["available"] = bool(snap.get("ready")) and out["camera"] == "ok"
        return out

    def state_dict(self) -> dict:
        with self._lock:
            eng = self._engine
            presence = self.presence.snapshot() if self.presence is not None else None
            return {
                "backend": self.backend,
                "has_depth": self.has_depth,
                "camera": self.camera,
                "camera_detail": self.camera_detail,
                "camera_waiting_s": self.camera_waiting_s,
                "generation": self.generation,
                "recycles": self.recycles,
                "idle": self.idle,
                "presence": presence,
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
