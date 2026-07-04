"""The gesture engine: turns hand landmarks into knob / swipe / fist events.

Design notes
------------
* Everything is time-based (no fixed frame-rate assumptions) and pure
  numpy/python, so it is unit-testable with synthetic landmark streams.

* **Knob**: engagement is a thumb–index pinch (like gripping a small knob),
  with hysteresis on the pinch ratio and a frame-count debounce. While
  engaged, rotation is measured as the median of the frame-to-frame angular
  deltas of four rigid hand vectors (wrist->middle-MCP, wrist->index-MCP,
  wrist->pinky-MCP, index-MCP->pinky-MCP). Using *deltas* instead of an
  absolute pose angle makes the estimate robust: the four vectors never need
  to agree on an absolute angle, only on how much the hand rotated this
  frame, and the median rejects any single landmark glitch. The accumulated
  angle runs through a One Euro filter (low lag when moving, steady at rest)
  and a small engage deadband so grabbing the knob never nudges the volume.

* **Ratchet regrip** falls out for free: release the pinch, rotate your hand
  back, pinch again — exactly like a physical knob, accumulation only
  happens while gripped.

* **Swipe**: open palm moving predominantly horizontally with enough travel
  and speed inside a sliding window. Swipes are suppressed while the knob is
  engaged and for `min_presence_s` after a hand first appears (so a hand
  entering the frame never skips a track).

* **Depth gating** (when the Kinect provides registered depth): hands outside
  the configured distance band are ignored entirely — people walking past in
  the background can't touch your volume.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import AppConfig
from ..filters import OneEuroFilter, wrap_deg
from ..types import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    DepthSampler,
    EngineSnapshot,
    FistHold,
    GestureEvent,
    Hand,
    KnobEngage,
    KnobRelease,
    KnobTurn,
    Swipe,
)

# Rigid hand vectors used for rotation estimation: (from_landmark, to_landmark)
_ROTATION_VECTORS = (
    (WRIST, MIDDLE_MCP),
    (WRIST, INDEX_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, PINKY_MCP),
)

_FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

_EXTENDED_FACTOR = 1.10   # tip must be this much farther from wrist than pip


def _vector_angles(pts: np.ndarray) -> np.ndarray:
    """Angles (deg) of the rotation-reference vectors, in image coords (y down,
    so positive delta = clockwise on screen)."""
    out = np.empty(len(_ROTATION_VECTORS), dtype=np.float64)
    for i, (a, b) in enumerate(_ROTATION_VECTORS):
        d = pts[b] - pts[a]
        out[i] = np.degrees(np.arctan2(d[1], d[0]))
    return out


def pinch_ratio(hand: Hand) -> float:
    size = hand.size
    if size < 1e-6:
        return 10.0
    return float(np.linalg.norm(hand.pts[THUMB_TIP] - hand.pts[INDEX_TIP])) / size


def openness(hand: Hand) -> str:
    """'open' (>=4 fingers extended), 'fist' (<=1), else 'neutral'."""
    wrist = hand.pts[WRIST]
    extended = 0
    for tip, pip in _FINGERS:
        if np.linalg.norm(hand.pts[tip] - wrist) > _EXTENDED_FACTOR * np.linalg.norm(hand.pts[pip] - wrist):
            extended += 1
    if extended >= 4:
        return "open"
    if extended <= 1:
        return "fist"
    return "neutral"


@dataclass
class _TrackedHand:
    hand: Hand
    depth_m: Optional[float]
    pinch: float
    pose: str


class GestureEngine:
    IDLE, ENGAGING, ENGAGED = "idle", "engaging", "engaged"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.state = self.IDLE

        # Primary-hand tracking (nearest-neighbour association across frames)
        self._last_palm: Optional[np.ndarray] = None
        self._hand_first_seen: float = 0.0
        self._hand_last_seen: float = 0.0

        # +1 when the view is mirrored (selfie view, the default): screen
        # clockwise == user clockwise and screen-right == user-right. -1 for an
        # unmirrored feed, so gesture semantics stay physically correct.
        self._view_sign = 1 if cfg.capture.mirror else -1

        # Knob state
        self._engage_count = 0
        self._release_count = 0
        self._prev_angles: Optional[np.ndarray] = None
        self._prev_angles_t = 0.0
        self._accum_deg = 0.0
        self._filter = OneEuroFilter(cfg.knob.filter_min_cutoff, cfg.knob.filter_beta)
        self._effective_deg = 0.0
        self._last_emitted_deg = 0.0

        # Motion history for swipes / palm speed: (t, x, y, pose)
        self._history: deque[tuple[float, float, float, str]] = deque(maxlen=90)
        self._swipe_block_until = 0.0

        # Fist-hold
        self._fist_since: Optional[float] = None
        self._fist_block_until = 0.0

        self._snapshot = EngineSnapshot()
        self._frame_w = 640
        self._frame_h = 480

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def update(
        self,
        hands: list[Hand],
        t: float,
        frame_w: int,
        frame_h: int,
        depth_sampler: Optional[DepthSampler] = None,
    ) -> list[GestureEvent]:
        self._frame_w, self._frame_h = frame_w, frame_h
        events: list[GestureEvent] = []
        snap = EngineSnapshot(state=self.state)

        tracked = self._select_primary(hands, t, depth_sampler, snap)

        if tracked is None:
            self._history.clear()
            self._fist_since = None
            if self.state == self.ENGAGED:
                if t - self._hand_last_seen > self.cfg.knob.hand_lost_grace_s:
                    events.append(KnobRelease(t=t, deg=self._effective_deg))
                    self._to_idle()
                    snap.last_event = "release (hand lost)"
                # else: keep gripping through the dropout. The angle reference
                # is re-based on reacquisition (gap check in _update_knob), so
                # rotation during the blind gap is ignored rather than guessed.
            else:
                self._to_idle()
            snap.state = self.state
            self._snapshot = snap
            return events

        self._hand_last_seen = t
        hand, pinch, pose = tracked.hand, tracked.pinch, tracked.pose
        palm = hand.palm_center
        self._history.append((t, float(palm[0]), float(palm[1]), pose))
        speed = self._palm_speed(t)

        snap.hand_present = True
        snap.handedness = hand.handedness
        snap.pinch_ratio = round(pinch, 3)
        snap.openness = pose
        snap.palm_xy = (float(palm[0]), float(palm[1]))
        snap.palm_speed = round(speed, 3)
        snap.hand_depth_m = tracked.depth_m

        events += self._update_knob(hand, pinch, speed, t, snap)
        if self.state != self.ENGAGED:
            events += self._update_swipe(t, snap)
            events += self._update_fist(pose, speed, t, snap)
        else:
            self._fist_since = None

        snap.state = self.state
        snap.angle_deg = round(self._effective_deg, 2)
        self._snapshot = snap
        return events

    def snapshot(self) -> EngineSnapshot:
        return self._snapshot

    # ------------------------------------------------------------------
    # primary hand selection + gating
    # ------------------------------------------------------------------
    def _select_primary(
        self,
        hands: list[Hand],
        t: float,
        depth_sampler: Optional[DepthSampler],
        snap: EngineSnapshot,
    ) -> Optional[_TrackedHand]:
        gate = self.cfg.gate
        candidates: list[_TrackedHand] = []
        for hand in hands:
            if hand.size < gate.min_hand_frac * self._frame_h:
                snap.gated_out = "too small / too far"
                continue
            depth_m: Optional[float] = None
            if depth_sampler is not None and gate.use_depth:
                depth_m = self._sample_hand_depth(hand, depth_sampler)
                if depth_m is not None and not (gate.depth_min_m <= depth_m <= gate.depth_max_m):
                    snap.gated_out = f"outside depth band ({depth_m:.2f} m)"
                    continue
            candidates.append(_TrackedHand(hand, depth_m, pinch_ratio(hand), openness(hand)))

        if not candidates:
            # While engaged, remember where the gripping hand was so that only
            # IT can resume the grip after a dropout — never someone else's hand.
            if self.state != self.ENGAGED:
                self._last_palm = None
            return None

        chosen: Optional[_TrackedHand] = None
        if self._last_palm is not None:
            # Stick with the hand we were tracking if it's still around.
            best_d = 0.30 * self._frame_w
            for c in candidates:
                d = float(np.linalg.norm(c.hand.palm_center - self._last_palm))
                if d < best_d:
                    best_d = d
                    chosen = c
        if chosen is None:
            if self.state == self.ENGAGED:
                # A different hand must not inherit an engaged knob (and its
                # accumulated rotation). Treat as "hand lost" instead.
                return None
            chosen = max(candidates, key=lambda c: c.hand.size)
            self._hand_first_seen = t
            self._history.clear()
        self._last_palm = chosen.hand.palm_center
        return chosen

    @staticmethod
    def _sample_hand_depth(hand: Hand, depth_sampler: DepthSampler) -> Optional[float]:
        readings = []
        for idx in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP):
            d = depth_sampler(float(hand.pts[idx][0]), float(hand.pts[idx][1]))
            if d is not None and d > 0.05:
                readings.append(d)
        if not readings:
            return None
        return float(np.median(readings))

    def _palm_speed(self, t: float, horizon: float = 0.12) -> float:
        """Mean palm speed over the last `horizon` seconds, in frame-widths/s."""
        pts = [(ht, hx, hy) for (ht, hx, hy, _) in self._history if t - ht <= horizon]
        if len(pts) < 2:
            return 0.0
        dt = pts[-1][0] - pts[0][0]
        if dt <= 1e-6:
            return 0.0
        dist = float(np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]))
        return dist / dt / self._frame_w

    # ------------------------------------------------------------------
    # knob
    # ------------------------------------------------------------------
    def _update_knob(
        self, hand: Hand, pinch: float, speed: float, t: float, snap: EngineSnapshot
    ) -> list[GestureEvent]:
        cfg = self.cfg.knob
        events: list[GestureEvent] = []
        angles = _vector_angles(hand.pts)

        if self.state == self.IDLE:
            if pinch < cfg.engage_pinch and speed < cfg.max_engage_speed:
                self.state = self.ENGAGING
                self._engage_count = 1
                self._prev_angles = angles
            return events

        if self.state == self.ENGAGING:
            if pinch < cfg.engage_pinch and speed < cfg.max_engage_speed:
                self._engage_count += 1
                self._prev_angles = angles  # warm the reference; rotation counts from grip
                if self._engage_count >= cfg.engage_frames:
                    self.state = self.ENGAGED
                    self._release_count = 0
                    self._accum_deg = 0.0
                    self._effective_deg = 0.0
                    self._last_emitted_deg = 0.0
                    self._filter.reset()
                    self._filter(t, 0.0)
                    events.append(KnobEngage(t=t))
                    snap.last_event = "knob engaged"
            else:
                self.state = self.IDLE
                self._prev_angles = None
            return events

        # ENGAGED
        if pinch > cfg.release_pinch:
            self._release_count += 1
            if self._release_count >= cfg.release_frames:
                events.append(KnobRelease(t=t, deg=self._effective_deg))
                snap.last_event = "knob released"
                self._to_idle()
                # Releasing the pinch flicks the landmarks around; block swipes
                # briefly so the release never reads as a swipe.
                self._swipe_block_until = t + 0.4
                return events
        else:
            self._release_count = 0

        # Re-base after a tracking gap (dropout / reacquisition): comparing
        # angles across a blind gap would apply the whole gap's rotation as one
        # frame delta — a jump. Skip accumulation for that frame instead.
        fresh_reference = self._prev_angles is not None and (t - self._prev_angles_t) <= 0.12
        if fresh_reference:
            deltas = np.array([wrap_deg(a - p) for a, p in zip(angles, self._prev_angles)])
            delta = float(np.median(deltas))
            if abs(delta) <= cfg.max_frame_delta_deg:
                self._accum_deg += delta
        self._prev_angles = angles
        self._prev_angles_t = t

        filtered = self._filter(t, self._accum_deg)
        if abs(filtered) <= cfg.deadband_deg:
            effective = 0.0
        else:
            effective = filtered - np.sign(filtered) * cfg.deadband_deg
        effective *= self._view_sign
        if cfg.invert:
            effective = -effective
        self._effective_deg = effective

        if abs(effective - self._last_emitted_deg) >= 0.2:
            events.append(
                KnobTurn(t=t, deg=effective, delta_deg=effective - self._last_emitted_deg)
            )
            self._last_emitted_deg = effective
        return events

    def _to_idle(self) -> None:
        self.state = self.IDLE
        self._engage_count = 0
        self._release_count = 0
        self._prev_angles = None
        self._accum_deg = 0.0
        self._effective_deg = 0.0
        self._last_emitted_deg = 0.0
        self._filter.reset()

    # ------------------------------------------------------------------
    # swipe
    # ------------------------------------------------------------------
    def _update_swipe(self, t: float, snap: EngineSnapshot) -> list[GestureEvent]:
        cfg = self.cfg.swipe
        if not cfg.enabled or t < self._swipe_block_until:
            return []
        if t - self._hand_first_seen < cfg.min_presence_s:
            return []

        window = [(ht, hx, hy, pose) for (ht, hx, hy, pose) in self._history if t - ht <= cfg.window_s]
        if len(window) < 4:
            return []
        # Nearly every sample in the window must be open-palm (allow one miss).
        if sum(1 for w in window if w[3] == "open") < len(window) - 1:
            return []
        duration = window[-1][0] - window[0][0]
        if duration < 0.12:
            return []
        dx = window[-1][1] - window[0][1]
        dy = window[-1][2] - window[0][2]
        travel = abs(dx) / self._frame_w
        speed = travel / duration
        if travel < cfg.min_travel_frac or speed < cfg.min_speed_frac:
            return []
        if abs(dy) > cfg.max_vertical_ratio * abs(dx):
            return []

        direction = (1 if dx > 0 else -1) * self._view_sign
        self._swipe_block_until = t + cfg.cooldown_s
        self._history.clear()
        snap.last_event = f"swipe {'right (next)' if direction > 0 else 'left (previous)'}"
        return [Swipe(t=t, direction=direction, speed=speed)]

    # ------------------------------------------------------------------
    # fist-hold (play/pause)
    # ------------------------------------------------------------------
    def _update_fist(self, pose: str, speed: float, t: float, snap: EngineSnapshot) -> list[GestureEvent]:
        cfg = self.cfg.fist
        if not cfg.enabled or t < self._fist_block_until:
            self._fist_since = None if not cfg.enabled else self._fist_since
            return []
        if pose == "fist" and speed < cfg.max_speed_frac:
            if self._fist_since is None:
                self._fist_since = t
            elif t - self._fist_since >= cfg.hold_s:
                self._fist_since = None
                self._fist_block_until = t + cfg.cooldown_s
                snap.last_event = "fist hold (play/pause)"
                return [FistHold(t=t)]
        else:
            self._fist_since = None
        return []
