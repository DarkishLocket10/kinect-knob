"""Depth-based presence: is a person actually in front of the camera?

Why this exists
---------------
Two consumers, opposite duty cycles:

* **kinect-knob itself** only needs the expensive half of its pipeline —
  MediaPipe hand landmarking, depth registration, the low-light boost — while
  somebody is there to gesture at it. An empty room ran the full 30 Hz stack
  around the clock on a GPU that has better things to do.
* **whiteboard-sync** needs the exact opposite question answered about a
  *region*: "is a human in front of this board right now", so it can refuse to
  photograph a board that a head is sitting in front of. Percentile stats over
  a whole board half (the old ``region_depth`` guard) miss a head — it covers
  a few percent of the region, so the 10th percentile never moves. Occupancy
  against a learned empty-scene background does not miss it.

Both are the same measurement, so it lives in one place and is served over
``/api/presence``.

How it works
------------
Classic depth background subtraction, which on a time-of-flight sensor is
unusually well behaved: the ToF stream self-illuminates, so this works in a
pitch-black room, and "distance to the wall" is a far more stable background
model than any colour-image equivalent.

* ``bg`` is a per-pixel model of the EMPTY scene (walls, desk, monitors).
* A pixel is foreground when it sits ``fg_gap_mm`` or more in FRONT of its
  background value and falls inside the depth band of interest.
* ``occupancy`` is the foreground fraction of valid pixels. A person standing
  in the frame reads tens of percent; a head leaning into a board region reads
  a few percent — hence the deliberately low default threshold.

The background only adapts while the scene reads EMPTY. That is the crucial
detail for the whiteboard: Yash sits nearly still at the work board for hours,
and a background that kept adapting would quietly absorb him into the wall and
declare the board clear. Adaptation while absent is still needed (a chair gets
moved, a monitor is raised), and it is fast in the "something went away"
direction and slow in the "something arrived" direction, so a scene that is
genuinely empty re-converges in seconds without a newly parked object being
swallowed instantly.

Hysteresis is asymmetric on purpose: presence latches on fast (a gesture user
must not wait for the pipeline to wake) and releases slowly (``linger_s``),
because somebody who stepped out of frame for a moment is still *there*, and a
whiteboard photo taken in that gap is exactly the photo that produces a
half-occluded line and a garbage Todoist task.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PresenceResult:
    """One probe's verdict. ``occupancy`` is the raw measurement; ``present``
    is that measurement after hysteresis and linger."""
    present: bool
    occupancy: float
    valid_frac: float
    nearest_mm: Optional[int]
    t: float


class DepthPresence:
    """Whole-frame presence with a learned empty-scene background.

    Thread-safety: ``update`` is called from one thread (the vision loop);
    ``snapshot`` and ``region`` are called from the web executor. The lock
    covers the published verdict and the background array swap, not the
    arithmetic — a region query reading a background that is one probe stale
    is harmless.
    """

    def __init__(
        self,
        near_m: float = 0.4,
        far_m: float = 5.0,
        fg_gap_mm: float = 250.0,
        on_frac: float = 0.02,
        off_frac: float = 0.010,
        linger_s: float = 20.0,
        warmup_s: float = 3.0,
        motion_mm: float = 60.0,
        motion_frac: float = 0.25,
        static_absorb_s: float = 300.0,
    ):
        self.near_mm = near_m * 1000.0
        self.far_mm = far_m * 1000.0
        self.fg_gap_mm = fg_gap_mm
        self.on_frac = on_frac
        self.off_frac = off_frac
        self.linger_s = linger_s
        self.warmup_s = warmup_s
        # What tells a person sitting still apart from a chair somebody
        # parked in view — see _adapt. motion_frac is measured against the
        # FOREGROUND, not the frame: ToF edge pixels flicker by metres all
        # over an empty scene, so a whole-frame motion count never settles
        # (measured on the live sensor 2026-09-17) and nothing would ever be
        # absorbed. Motion inside the object being judged is the real signal.
        #
        # Calibration, same sensor and date: an EMPTY room's residual
        # foreground is pure speckle and reads moved_frac ~0.6, because
        # speckle by definition never sits still. A solid object's interior
        # pixels do sit still, so it reads far lower. Hence a threshold of
        # 0.25 — comfortably under the empty-room floor, comfortably over a
        # parked chair — and a long window on top, because "has not moved a
        # single time in five minutes" is the claim being made. The live
        # reading is `moved_frac` in /api/presence; field-tune against it
        # rather than re-deriving it.
        self.motion_mm = motion_mm
        self.motion_frac = motion_frac
        self.static_absorb_s = static_absorb_s

        self._lock = threading.Lock()
        self._bg: Optional[np.ndarray] = None
        self._bg_shape: Optional[tuple] = None
        self._started = time.monotonic()
        self._present = False
        self._last_seen = 0.0          # monotonic; last probe that read occupied
        self._last: Optional[PresenceResult] = None
        self._probes = 0
        self._prev: Optional[np.ndarray] = None   # previous probe's depth frame
        self._static_since: Optional[float] = None
        self._moved_frac = 0.0                   # live, for field tuning

    def configure(self, cfg) -> None:
        """Re-read the live PresenceConfig.

        The dashboard's tuning writes straight onto the shared AppConfig and
        every other parameter in this app is read fresh per frame, so a
        detector holding construction-time copies would present sliders that
        silently do nothing. Called once per probe — a handful of attribute
        writes, and cheaper than the alternative of threading the config
        through every call site.
        """
        self.near_mm = cfg.near_m * 1000.0
        self.far_mm = cfg.far_m * 1000.0
        self.fg_gap_mm = cfg.fg_gap_m * 1000.0
        self.on_frac = cfg.on_frac
        self.off_frac = cfg.off_frac
        self.linger_s = cfg.linger_s
        self.static_absorb_s = cfg.static_absorb_s

    # -- background -----------------------------------------------------
    def _adapt(self, depth_mm: np.ndarray, valid: np.ndarray, occupied: bool,
               static: bool) -> None:
        """Move the empty-scene model toward the current frame.

        Two directions, deliberately asymmetric:

        * **Farther** — whatever was here has gone. Always safe, always fast.
        * **Nearer** — something has arrived. Absorbing that is how a chair
          somebody parked in view stops reading as a person forever; doing it
          while a *person* is there is how the whiteboard guard quietly fails,
          because Yash sits nearly still at the work board for hours and the
          wall would grow around him.

        The discriminator is motion, not time alone: a person is never
        perfectly static at this resolution (they breathe, they shift, their
        outline flickers), furniture is. So the nearer direction only runs
        once the scene has been unchanged for ``static_absorb_s`` — and any
        motion at all resets that clock.
        """
        bg = self._bg
        if bg is None:
            return
        farther = valid & (depth_mm > bg)
        if farther.any():
            bg[farther] += 0.25 * (depth_mm[farther] - bg[farther])
        if occupied and not static:
            return
        rate = 0.02 if not occupied else 0.01
        nearer = valid & (depth_mm < bg)
        if nearer.any():
            bg[nearer] += rate * (depth_mm[nearer] - bg[nearer])

    def _static_for(self, d: np.ndarray, fg: np.ndarray, now: float) -> float:
        """Fraction of the current FOREGROUND that moved since the last probe.

        Returns the measurement and updates the static clock as a side effect.
        Frame-wide motion is useless here: a time-of-flight sensor flickers by
        whole metres along every depth discontinuity, so an empty room never
        reads still. What separates a person from a chair is whether the
        candidate object itself is moving.
        """
        prev = self._prev
        self._prev = d
        n_fg = int(fg.sum())
        if prev is None or prev.shape != d.shape or n_fg == 0:
            self._static_since = now
            self._moved_frac = 0.0 if n_fg == 0 else 1.0
            return self._moved_frac
        moved = fg & (np.abs(d - prev) > self.motion_mm)
        frac = float(moved.sum()) / n_fg
        self._moved_frac = frac
        if frac > self.motion_frac:
            self._static_since = now
        elif self._static_since is None:
            self._static_since = now
        return frac

    # -- probing --------------------------------------------------------
    def update(self, depth_mm: Optional[np.ndarray], now: Optional[float] = None) -> PresenceResult:
        """Run one probe against a depth frame (mm, 0 = no return).

        Cheap enough to call at a few Hz on a proc-sized map; deliberately NOT
        called per-frame at 30 Hz, since presence changes on human timescales.
        """
        now = time.monotonic() if now is None else now
        if depth_mm is None or depth_mm.size == 0:
            return self._publish(0.0, 0.0, None, now, measured=False)

        d = np.asarray(depth_mm, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        valid = np.isfinite(d) & (d > 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return self._publish(0.0, 0.0, None, now, measured=False)
        valid_frac = n_valid / d.size
        # The sensor emits NaN and +/-inf for pixels with no return. Every
        # mask below excludes them, so results were already correct — but the
        # frame-to-frame subtraction in _static_for runs before masking, and
        # inf - inf is a NaN and a RuntimeWarning per probe. Normalise once
        # here instead of defending in four places.
        d = np.where(valid, d, 0.0)

        with self._lock:
            if self._bg is None or self._bg_shape != d.shape:
                # Seed the model with the current frame. Anything standing
                # there at startup is therefore "background" until it moves —
                # unavoidable without an empty-room calibration pass, and it
                # self-corrects within seconds of the person moving.
                self._bg = np.where(valid, d, self.far_mm).astype(np.float32)
                self._bg_shape = d.shape
                self._started = now
            bg = self._bg

        in_band = valid & (d >= self.near_mm) & (d <= self.far_mm)
        fg = in_band & ((bg - d) >= self.fg_gap_mm)
        occupancy = float(fg.sum()) / max(n_valid, 1)
        nearest = None
        if fg.any():
            nearest = int(d[fg].min())

        occupied = occupancy >= (self.off_frac if self._present else self.on_frac)
        with self._lock:
            self._static_for(d, fg, now)
            static = (self._static_since is not None
                      and (now - self._static_since) >= self.static_absorb_s)
            self._adapt(d, valid, occupied, static)
            self._probes += 1
        return self._publish(occupancy, valid_frac, nearest, now,
                             measured=True, occupied=occupied)

    def _publish(self, occupancy: float, valid_frac: float, nearest: Optional[int],
                 now: float, measured: bool, occupied: bool = False) -> PresenceResult:
        with self._lock:
            warming = (now - self._started) < self.warmup_s
            if measured and occupied and not warming:
                self._last_seen = now
                self._present = True
            elif self._present and (now - self._last_seen) >= self.linger_s:
                self._present = False
            res = PresenceResult(present=self._present, occupancy=round(occupancy, 5),
                                 valid_frac=round(valid_frac, 3), nearest_mm=nearest,
                                 t=now)
            self._last = res
            return res

    # -- readers --------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            res = self._last
            last_seen = self._last_seen
            present = self._present
            ready = self._bg is not None
            probes = self._probes
            static_since = self._static_since
            moved_frac = self._moved_frac
        now = time.monotonic()
        return {
            "present": present,
            "ready": ready,
            "probes": probes,
            "since_seen_s": None if not last_seen else round(now - last_seen, 1),
            "occupancy": res.occupancy if res else 0.0,
            "valid_frac": res.valid_frac if res else 0.0,
            "nearest_mm": res.nearest_mm if res else None,
            "age_s": round(now - res.t, 1) if res else None,
            "static_s": None if static_since is None
                        else round(now - static_since, 1),
            "moved_frac": round(moved_frac, 4),
            "linger_s": self.linger_s,
            "on_frac": self.on_frac,
            "off_frac": self.off_frac,
        }

    @property
    def present(self) -> bool:
        with self._lock:
            return self._present

    def force_relearn(self) -> None:
        """Drop the background model; the next probe reseeds from the live
        frame. For the dashboard, after the furniture moves."""
        with self._lock:
            self._bg = None
            self._bg_shape = None
            self._present = False
            self._last_seen = 0.0
            self._prev = None
            self._static_since = None
            self._moved_frac = 0.0


class RegionPresence:
    """Occupancy of arbitrary regions of the FULL-RES colour-aligned depth map.

    Separate from :class:`DepthPresence` because it answers a different
    question against a different array: not "is anyone in the room" on the
    small tracking map, but "is something in front of THIS rectangle" on the
    1920-wide aligned map, in the same coordinates ``/api/snapshot`` serves.
    That is what lets whiteboard-sync ask about one board half — and it is the
    measurement the old percentile guard could not make, because a head covers
    a few percent of a board half and never moves a 10th percentile.

    Everything runs on a ``decim``-times decimated copy: a head is ~100x150 px
    at 1080p, so at 1/4 scale it is still ~25x37 px of evidence, while the
    background update costs ~1 ms instead of ~25 ms. Updates are additionally
    rate-limited — the board plane does not move.

    The background is the board plane itself, learned the same way as
    :class:`DepthPresence` and, likewise, frozen in the "something arrived"
    direction so that somebody sitting still in front of a board never fades
    into it.
    """

    def __init__(self, fg_gap_mm: float = 250.0, decim: int = 4,
                 min_interval_s: float = 1.0, motion_mm: float = 60.0,
                 motion_frac: float = 0.25, static_absorb_s: float = 900.0):
        self.fg_gap_mm = fg_gap_mm
        self.decim = max(1, int(decim))
        self.min_interval_s = min_interval_s
        # Same furniture-vs-person discriminator as DepthPresence, with a
        # longer patience: this model guards whiteboard photographs, and the
        # person it has to keep seeing is somebody working at a desk.
        self.motion_mm = motion_mm
        self.motion_frac = motion_frac
        self.static_absorb_s = static_absorb_s
        self._lock = threading.Lock()
        self._bg: Optional[np.ndarray] = None       # decimated empty-scene plane
        self._cur: Optional[np.ndarray] = None      # decimated latest frame
        self._shape: Optional[tuple] = None         # FULL-res shape it came from
        self._t = 0.0
        self._next_update = 0.0
        self._static_since: Optional[float] = None

    def _small(self, arr: np.ndarray) -> np.ndarray:
        k = self.decim
        return np.ascontiguousarray(arr[::k, ::k], dtype=np.float32)

    def update(self, aligned_mm: Optional[np.ndarray], now: Optional[float] = None) -> None:
        """Feed the latest full-res aligned depth map. Called from the capture
        thread at whatever rate registration happens to run; rate-limited here
        so an active-pipeline 15 Hz registration does not pay for it."""
        if aligned_mm is None or aligned_mm.size == 0:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            if now < self._next_update and self._bg is not None:
                return
            self._next_update = now + self.min_interval_s
        full_shape = aligned_mm.shape
        d = self._small(np.asarray(aligned_mm))
        valid = np.isfinite(d) & (d > 0)
        if not valid.any():
            return
        d = np.where(valid, d, 0.0)   # see DepthPresence.update
        with self._lock:
            if self._bg is None or self._shape != full_shape:
                self._bg = np.where(valid, d, 0.0).astype(np.float32)
                self._shape = full_shape
                self._cur = d
                self._t = time.time()
                return
            bg = self._bg
            # Pixels that have never returned depth take the first value they get.
            fresh = valid & (bg <= 0)
            if fresh.any():
                bg[fresh] = d[fresh]
            farther = valid & (bg > 0) & (d > bg)
            if farther.any():
                bg[farther] += 0.25 * (d[farther] - bg[farther])
            # Something NEARER than the plane is an object or a person, and
            # the two need opposite treatment: a box left on the shelf should
            # become part of the board, somebody standing at it must never.
            # Motion decides — the plane only grows around things that have
            # not moved at all for static_absorb_s.
            fg_now = valid & (bg > 0) & ((bg - d) >= self.fg_gap_mm)
            static = self._static_locked(d, fg_now, self._cur, now)
            if static:
                nearer = valid & (bg > 0) & (d < bg)
                if nearer.any():
                    bg[nearer] += 0.05 * (d[nearer] - bg[nearer])
            self._cur = d
            self._t = time.time()

    def _static_locked(self, d: np.ndarray, fg: np.ndarray,
                       prev: Optional[np.ndarray], now: float) -> bool:
        """Caller holds the lock. Has whatever is standing in front of this
        plane been motionless long enough to be treated as furniture?

        Measured against the FOREGROUND, not the whole region: see
        DepthPresence._static_for — on a ToF sensor, frame-wide motion counts
        never settle.
        """
        n_fg = int(fg.sum())
        if prev is None or prev.shape != d.shape or n_fg == 0:
            self._static_since = now
            return False
        moved = fg & (np.abs(d - prev) > self.motion_mm)
        if float(moved.sum()) / n_fg > self.motion_frac:
            self._static_since = now
            return False
        if self._static_since is None:
            self._static_since = now
        return (now - self._static_since) >= self.static_absorb_s

    def occupancy(self, x1: int, y1: int, x2: int, y2: int,
                  flip_x: bool = True) -> Optional[dict]:
        """Foreground fraction over a region of the latest aligned depth map.

        Coordinates are UNMIRRORED full-res colour coordinates — the frame
        ``/api/snapshot`` serves — so a whiteboard crop region passes through
        verbatim. ``flip_x`` maps them onto the raw (mirrored) depth array,
        exactly as ``KinectV2Capture.region_depth`` does.
        """
        with self._lock:
            bg, cur, shape, t = self._bg, self._cur, self._shape, self._t
        if bg is None or cur is None or shape is None:
            return None
        fh, fw = shape[:2]
        x1, x2 = sorted((max(0, min(fw, int(x1))), max(0, min(fw, int(x2)))))
        y1, y2 = sorted((max(0, min(fh, int(y1))), max(0, min(fh, int(y2)))))
        if x2 <= x1 or y2 <= y1:
            return None
        if flip_x:
            x1, x2 = fw - x2, fw - x1
        k = self.decim
        sh, sw = cur.shape[:2]
        sx1, sx2 = max(0, x1 // k), min(sw, -(-x2 // k))
        sy1, sy2 = max(0, y1 // k), min(sh, -(-y2 // k))
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        c = cur[sy1:sy2, sx1:sx2]
        r = bg[sy1:sy2, sx1:sx2]
        valid = np.isfinite(c) & (c > 0) & (r > 0)
        n_valid = int(valid.sum())
        age = round(time.time() - t, 1)
        if n_valid == 0:
            return {"occupancy": 0.0, "valid_frac": 0.0, "plane_mm": None,
                    "nearest_mm": None, "pixels": int(c.size), "age_s": age}
        fg = valid & ((r - c) >= self.fg_gap_mm)
        return {
            "occupancy": round(float(fg.sum()) / n_valid, 5),
            "valid_frac": round(n_valid / c.size, 3),
            "plane_mm": int(np.median(r[valid])),
            "nearest_mm": int(c[fg].min()) if fg.any() else None,
            "pixels": int(c.size),
            "age_s": age,
        }

    def force_relearn(self) -> None:
        with self._lock:
            self._bg = None
            self._shape = None
            self._cur = None
            self._next_update = 0.0
            self._static_since = None
