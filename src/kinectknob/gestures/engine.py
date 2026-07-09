"""The gesture engine: turns hand landmarks into knob / swipe / play-pause events.

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
  entering the frame never skips a track). Fast swipes motion-blur the hand
  enough for tracking to drop it for a frame or two mid-gesture, so brief
  dropouts (`gate.lost_grace_s`) keep the hand's identity, presence clock and
  motion history alive instead of resetting them.

* **Depth gating** (when the Kinect provides registered depth): hands outside
  the configured distance band are ignored entirely — people walking past in
  the background can't touch your volume.

* **Busy-hand rejection** (also depth-based): a hand holding an object (water
  bottle, mug, toothbrush) shows a surface over its palm area sitting well in
  FRONT of the wrist plane (object_gap). Such a hand is "busy": it cannot
  engage the knob, swipe, or toggle playback — and if the other hand is
  visible and free, primary tracking hands over to it, so the free hand
  controls the knob while the busy one keeps holding. The verdict lingers
  briefly (busy_linger_s) so depth flicker can't sneak an engage through.
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
    GestureEvent,
    Hand,
    KnobEngage,
    KnobRelease,
    KnobTurn,
    PlayPauseHold,
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


def curl_gap(hand: Hand) -> float:
    """Thumb-tip to middle-tip distance / hand size. In a hand relaxed into a
    curl ALL the fingertips cluster around the thumb, so this is small; in a
    deliberate pinch the middle finger hangs back from the pinching pair, so
    it stays large even when the other fingers are curled. This is what
    separates 'resting hand that happens to look like a pinch' from a real
    grip — openness() can't, because a natural pinch curls its spare fingers
    and classifies as a fist too."""
    size = hand.size
    if size < 1e-6:
        return 10.0
    return float(np.linalg.norm(hand.pts[THUMB_TIP] - hand.pts[MIDDLE_TIP])) / size


def openness(hand: Hand) -> str:
    """'open' (>=4 fingers extended), 'two' (index+middle only, the swipe
    pose), 'fist' (<=1 extended), else 'neutral'. Thumb is deliberately
    ignored throughout — people fold or splay it unpredictably."""
    wrist = hand.pts[WRIST]
    ext = []
    for tip, pip in _FINGERS:
        ext.append(
            np.linalg.norm(hand.pts[tip] - wrist)
            > _EXTENDED_FACTOR * np.linalg.norm(hand.pts[pip] - wrist)
        )
    if sum(ext) >= 4:
        return "open"
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        return "two"
    if sum(ext) <= 1:
        return "fist"
    return "neutral"


def palm_facing_score(hand: Hand) -> float:
    """How much the palm faces the camera: ~+0.8 for a flat palm toward the
    lens, ~-0.8 for the back of the hand, ~0 edge-on. The 2D cross product of
    wrist->index-MCP x wrist->pinky-MCP, normalised by hand size squared, with
    the sign folded through MediaPipe handedness. View-mirroring flips both
    the cross product and the reported handedness, so the score is invariant
    to whether the feed is mirrored.

    Sign convention FIELD-VERIFIED 2026-07-07 on the live pipeline (mirrored
    feed + this MediaPipe build): the back of a raised right hand was scoring
    positive under the opposite sign, triggering play/pause. If your camera or
    MediaPipe version behaves differently, flip ``playpause.invert_facing``
    instead of editing this."""
    size = hand.size
    if size < 1e-6:
        return 0.0
    a = hand.pts[INDEX_MCP] - hand.pts[WRIST]
    b = hand.pts[PINKY_MCP] - hand.pts[WRIST]
    cross = float(a[0] * b[1] - a[1] * b[0])
    return (cross if hand.handedness == "Left" else -cross) / (size * size)


def finger_spread(hand: Hand) -> float:
    """Mean gap between adjacent fingertips (index..pinky) over hand size.
    A deliberate open palm spreads its fingers (~0.4); a hand wrapped around
    or pressed flat against a held object bunches them (~0.15)."""
    size = hand.size
    if size < 1e-6:
        return 0.0
    tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    gaps = [np.linalg.norm(hand.pts[a] - hand.pts[b]) for a, b in zip(tips, tips[1:])]
    return float(np.mean(gaps)) / size


def object_gap(hand: Hand, depth_sampler: DepthSampler) -> Optional[float]:
    """Metres by which the nearest surface over the palm area sits in FRONT of
    the wrist. An empty hand reads near zero in any orientation (the palm-area
    pixels land on the hand itself); a hand wrapped around a bottle / mug /
    phone puts the object's surface well in front of the wrist plane. This is
    object EVIDENCE, deliberately not hand shape — the knob pinch is shaped
    exactly like holding a small object, so shape alone can't gate it.
    None when depth has no readings here."""
    d_wrist = depth_sampler(float(hand.pts[WRIST][0]), float(hand.pts[WRIST][1]))
    if d_wrist is None or d_wrist <= 0.05:
        return None
    palm = hand.palm_center
    probes = (
        palm,
        (palm + hand.pts[INDEX_MCP]) / 2.0,
        (palm + hand.pts[PINKY_MCP]) / 2.0,
    )
    depths = []
    for p in probes:
        d = depth_sampler(float(p[0]), float(p[1]))
        if d is not None and d > 0.05:
            depths.append(d)
    if not depths:
        return None
    return float(d_wrist - min(depths))


@dataclass
class _TrackedHand:
    hand: Hand
    depth_m: Optional[float]
    pinch: float
    curl: float
    pose: str
    holding: bool = False
    obj_gap: Optional[float] = None


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

        # Play/pause hold (open palm facing the camera, or fist)
        self._pp_since: Optional[float] = None
        self._pp_block_until = 0.0

        # Busy hand (holding an object): verdict lingers past the last holding
        # frame so depth flicker can't let a bottle hand briefly grab the knob.
        self._busy_until = 0.0

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
            self._pp_since = None
            lost_for = t - self._hand_last_seen
            if self.state == self.ENGAGED:
                self._history.clear()
                if lost_for > self.cfg.knob.hand_lost_grace_s:
                    events.append(KnobRelease(t=t, deg=self._effective_deg))
                    self._to_idle()
                    snap.last_event = "release (hand lost)"
                # else: keep gripping through the dropout. The angle reference
                # is re-based on reacquisition (gap check in _update_knob), so
                # rotation during the blind gap is ignored rather than guessed.
            elif lost_for > self.cfg.gate.lost_grace_s:
                self._history.clear()
                self._to_idle()
            # else: a dropout too brief to mean the hand left — typically ONE
            # motion-blurred frame in the middle of a fast swipe. Keep the
            # swipe history, presence clock and hand identity alive so the
            # gesture completes when tracking reacquires; clearing here used
            # to make every fast (= blurry) swipe physically impossible.
            snap.state = self.state
            self._snapshot = snap
            return events

        self._hand_last_seen = t
        hand, pinch, curl, pose = tracked.hand, tracked.pinch, tracked.curl, tracked.pose
        palm = hand.palm_center
        self._history.append((t, float(palm[0]), float(palm[1]), pose))
        speed = self._palm_speed(t)

        snap.hand_present = True
        snap.handedness = hand.handedness
        snap.pinch_ratio = round(pinch, 3)
        snap.curl_gap = round(curl, 3)
        snap.openness = pose
        snap.palm_xy = (float(palm[0]), float(palm[1]))
        snap.palm_speed = round(speed, 3)
        snap.hand_depth_m = tracked.depth_m

        # A hand holding an object is "busy": it can't grab the knob, swipe,
        # or toggle playback — use the other hand for control instead.
        if tracked.holding:
            self._busy_until = t + self.cfg.gate.busy_linger_s
        busy = tracked.holding or t < self._busy_until
        if tracked.obj_gap is not None:
            snap.extra["obj_gap"] = round(tracked.obj_gap, 3)
        if tracked.obj_gap is not None or busy:
            snap.extra["holding"] = busy

        events += self._update_knob(hand, pinch, curl, pose, speed, t, snap, busy)
        if self.state != self.ENGAGED:
            if busy:
                self._pp_since = None
            else:
                events += self._update_swipe(t, snap)
                events += self._update_playpause(hand, pose, speed, t, snap, depth_sampler)
        else:
            self._pp_since = None

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
            # Sticky confidence: MediaPipe's score routinely dips while fingers
            # occlude each other mid-pinch. The hand we are ALREADY tracking
            # keeps the benefit of the doubt at half the threshold; the full
            # bar applies only to admitting a new hand (ghost filtering).
            min_score = gate.min_score
            if self._last_palm is not None and float(
                np.linalg.norm(hand.palm_center - self._last_palm)
            ) < 0.30 * self._frame_w:
                min_score *= 0.5
            if hand.score < min_score:
                snap.gated_out = f"low confidence ({hand.score:.2f})"
                continue
            if hand.size < gate.min_hand_frac * self._frame_h:
                snap.gated_out = "too small / too far"
                continue
            depth_m: Optional[float] = None
            gap: Optional[float] = None
            holding = False
            if depth_sampler is not None and gate.use_depth:
                depth_m = self._sample_hand_depth(hand, depth_sampler)
                if depth_m is not None and not (gate.depth_min_m <= depth_m <= gate.depth_max_m):
                    snap.gated_out = f"outside depth band ({depth_m:.2f} m)"
                    continue
                if gate.object_gap_m > 0:
                    gap = object_gap(hand, depth_sampler)
                    holding = gap is not None and gap > gate.object_gap_m
            candidates.append(
                _TrackedHand(
                    hand, depth_m, pinch_ratio(hand), curl_gap(hand), openness(hand),
                    holding, gap,
                )
            )

        if not candidates:
            # Keep the identity anchor while ENGAGED (only the gripping hand
            # may resume the grip) and through brief dropouts (so a blurred
            # frame mid-swipe doesn't demote the hand to a stranger, which
            # would reset the swipe presence clock). Drop it once truly gone.
            if self.state != self.ENGAGED and (
                t - self._hand_last_seen > gate.lost_grace_s
            ):
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
        free = [c for c in candidates if not c.holding]
        if (
            chosen is not None and chosen.holding and free
            and self.state != self.ENGAGED
        ):
            # The tracked hand is wrapped around an object and a free hand is
            # up: hand control over to the free one (knob with the other hand).
            chosen = None
        if chosen is None:
            if self.state == self.ENGAGED:
                # A different hand must not inherit an engaged knob (and its
                # accumulated rotation). Treat as "hand lost" instead.
                return None
            # Prefer a free hand over one holding an object, whatever the size.
            chosen = max(free or candidates, key=lambda c: c.hand.size)
            self._hand_first_seen = t
            self._history.clear()
            self._busy_until = 0.0  # the busy linger belonged to the old hand
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
        self, hand: Hand, pinch: float, curl: float, pose: str, speed: float,
        t: float, snap: EngineSnapshot, busy: bool = False
    ) -> list[GestureEvent]:
        cfg = self.cfg.knob
        events: list[GestureEvent] = []
        angles = _vector_angles(hand.pts)

        # A relaxed/curled hand reads as a pinch too (the thumb naturally rests
        # against the curled index) — the classic false engage. But a real
        # pinch ALSO curls its spare fingers and classifies as a "fist", so
        # pose alone must not block. The discriminator is the curl gap: in a
        # resting curl every fingertip clusters near the thumb; in a real
        # pinch the middle finger hangs back. Block only when both agree.
        # Neither is re-checked while engaged — release is the pinch's job.
        pinch_ok = pinch < cfg.engage_pinch and speed < cfg.max_engage_speed
        relaxed_curl = (
            cfg.curl_reject_gap > 0 and pose == "fist" and curl < cfg.curl_reject_gap
        )

        if self.state == self.IDLE:
            if pinch_ok and not relaxed_curl and not busy:
                self.state = self.ENGAGING
                self._engage_count = 1
                self._prev_angles = angles
            return events

        if self.state == self.ENGAGING:
            if busy:
                # The hand turned out to be holding an object (pinching a
                # bottle / toothbrush reads exactly like the knob grip): abort.
                # An already-ENGAGED grip is never busy-checked — release stays
                # the pinch's job, consistent with the pose/curl rule above.
                self.state = self.IDLE
                self._prev_angles = None
                return events
            if pinch_ok and relaxed_curl:
                # Pose/curl flicker mid-debounce (fingers mid-curl during a
                # regrip): HOLD the count rather than resetting to idle, or
                # rapid ratchet regrips rarely survive the debounce window.
                self._prev_angles = angles
                return events
            if pinch_ok:
                self._engage_count += 1
                self._prev_angles = angles  # warm the reference; rotation counts from grip
                if self._engage_count >= cfg.engage_frames:
                    self.state = self.ENGAGED
                    self._release_count = 0
                    self._accum_deg = 0.0
                    self._effective_deg = 0.0
                    self._last_emitted_deg = 0.0
                    # Rebuilt (not just reset) so live-tuned smoothing params
                    # take effect on the next grip without a restart.
                    self._filter = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
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
        # The swipe pose must hold through (most of) the motion. Two-finger
        # mode is distinctive enough — an accidental match is rare — that we
        # can tolerate 25% misclassified samples (fast motion blurs the
        # landmarks); open-palm mode keeps the stricter allow-one-miss rule
        # because open hands occur constantly in natural movement.
        want = "two" if cfg.two_finger else "open"
        needed = max(3, (3 * len(window)) // 4) if cfg.two_finger else len(window) - 1
        if sum(1 for w in window if w[3] == want) < needed:
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
        if cfg.invert:
            direction = -direction
        self._swipe_block_until = t + cfg.cooldown_s
        # A palm lingering at the end of an open-palm swipe must not read as
        # a play/pause hold — block it for its own cooldown.
        self._pp_since = None
        self._pp_block_until = max(
            self._pp_block_until, t + self.cfg.playpause.cooldown_s)
        self._history.clear()
        snap.last_event = f"swipe {'right (next)' if direction > 0 else 'left (previous)'}"
        return [Swipe(t=t, direction=direction, speed=speed)]

    # ------------------------------------------------------------------
    # play/pause hold
    # ------------------------------------------------------------------
    def _update_playpause(
        self,
        hand: Hand,
        pose: str,
        speed: float,
        t: float,
        snap: EngineSnapshot,
        depth_sampler: Optional[DepthSampler] = None,
    ) -> list[GestureEvent]:
        """Play/pause on a held pose. Default pose is an open palm that must
        actually FACE the camera (palm_facing_score) — a hand waving past or
        the back of a raised hand never toggles playback. A hand HOLDING
        something is rejected two ways: bunched fingertips (finger_spread)
        and, with depth available, an object surface sitting closer to the
        camera than the wrist plane. "fist" mode keeps the old behaviour."""
        cfg = self.cfg.playpause
        if not cfg.enabled or t < self._pp_block_until:
            self._pp_since = None
            return []
        if t - self._hand_first_seen < cfg.min_presence_s:
            return []  # a hand entering the frame must not instantly toggle
        if cfg.pose == "palm":
            held = pose == "open"
            if held:
                facing = palm_facing_score(hand)
                if cfg.invert_facing:
                    facing = -facing
                spread = finger_spread(hand)
                snap.extra["facing"] = round(facing, 2)
                snap.extra["spread"] = round(spread, 2)
                if cfg.require_facing and facing < cfg.facing_min:
                    held = False
                if held and cfg.spread_min > 0 and spread < cfg.spread_min:
                    held = False  # fingers bunched: likely wrapped around something
                if held and cfg.object_gap_m > 0 and depth_sampler is not None:
                    palm = hand.palm_center
                    d_palm = depth_sampler(float(palm[0]), float(palm[1]))
                    d_wrist = depth_sampler(
                        float(hand.pts[WRIST][0]), float(hand.pts[WRIST][1]))
                    if d_palm is not None and d_wrist is not None:
                        gap = d_wrist - d_palm
                        snap.extra["obj_gap"] = round(gap, 3)
                        if gap > cfg.object_gap_m:
                            held = False  # something sits in front of the palm
        else:
            held = pose == "fist"
        if held and speed < cfg.max_speed_frac:
            if self._pp_since is None:
                self._pp_since = t
            elif t - self._pp_since >= cfg.hold_s:
                self._pp_since = None
                self._pp_block_until = t + cfg.cooldown_s
                snap.last_event = ("open palm" if cfg.pose == "palm" else "fist") + \
                    " hold (play/pause)"
                return [PlayPauseHold(t=t, pose=cfg.pose)]
        else:
            self._pp_since = None
        return []
