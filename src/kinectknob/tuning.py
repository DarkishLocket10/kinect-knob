"""Live tuning: user-adjustable gesture/volume parameters, editable from the
dashboard while the app runs.

Design:

* ``TUNABLES`` is the single source of truth — key, range, step, unit and a
  plain-English explanation for every knob the dashboard exposes. The web UI
  renders itself from this schema, so adding a tunable here is the whole job.
* Every listed parameter is read fresh each frame/event by the engine or
  controller, so writes to the shared ``AppConfig`` apply instantly — no
  restart. (The One Euro filter is rebuilt on each knob engage for the same
  reason.)
* Values changed away from their baseline are saved as a small JSON overlay
  (``KK_TUNING_PATH``, a mounted volume) and re-applied on startup, layered on
  top of defaults + yaml + env. Untouched keys keep following ``.env``.
* Safety rails: values are clamped to their published range, and paired
  thresholds (engage/release pinch, near/far distance) are kept a sane gap
  apart so no slider combination can wedge the engine.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig

log = logging.getLogger("kk.tune")

_DEFAULT_PATH = "/app/data/tuning.json"


@dataclass(frozen=True)
class Tunable:
    key: str        # dotted path on AppConfig, e.g. "knob.engage_pinch"
    label: str
    help: str
    kind: str       # float | int | bool
    group: str
    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    unit: str = ""  # shown after the value: "°", "s", "m", "frames", "w/s", "%"


# (lower_key, upper_key, minimum_gap): keep threshold pairs from crossing.
PAIRED = (
    ("knob.engage_pinch", "knob.release_pinch", 0.05),
    ("gate.depth_min_m", "gate.depth_max_m", 0.2),
)

TUNABLES: tuple[Tunable, ...] = (
    # ------------------------------------------------ volume limits & feel
    Tunable(
        "ha.max_volume", "Volume ceiling",
        "Gestures can never set the volume above this. The Bose remote and app "
        "can still go higher — only gesture control is capped. If the volume is "
        "already above the ceiling, any grip pulls it back under.",
        "float", "Volume limits & feel", 0.05, 1.0, 0.01, "%",
    ),
    Tunable(
        "knob.full_scale_deg", "Twist for full volume sweep",
        "Total hand rotation to sweep the volume from 0% to 100%. Higher means "
        "finer control and more re-grips to make a big change. At 1800°, one "
        "percent of volume is 18° of twist, so with a 20% ceiling bottom-to-top "
        "takes about two full ratcheted grips.",
        "float", "Volume limits & feel", 180, 3600, 30, "°",
    ),
    Tunable(
        "ha.volume_step", "Fallback step size",
        "Only used before the soundbar has reported its volume (relative mode): "
        "each detent of twist nudges the volume by this much. Upward nudges in "
        "this mode stop one step below the ceiling and never fire blind.",
        "float", "Volume limits & feel", 0.005, 0.10, 0.005, "%",
    ),
    Tunable(
        "ha.send_interval_s", "Volume update rate",
        "Minimum gap between volume commands sent to the soundbar while you "
        "twist. Lower = the dial tracks your hand more tightly (more "
        "responsive); too low can flood Home Assistant or the Bose and "
        "actually add lag. 0.05 s (20 updates/s) is a good aggressive value.",
        "float", "Volume limits & feel", 0.03, 0.30, 0.01, "s",
    ),
    Tunable(
        "knob.invert", "Invert rotation",
        "Flip which twist direction means louder. If clockwise turns the volume "
        "down, toggle this.",
        "bool", "Volume limits & feel",
    ),
    Tunable(
        "capture.proc_width", "Tracking resolution",
        "Frame width (pixels) the hand tracker runs on. Lower = every frame "
        "processes faster and latency drops, but small/distant hands get "
        "harder to see. Applies to new frames immediately when lowering; "
        "raising it above the value the app started with needs a container "
        "restart to take full effect.",
        "int", "Performance", 384, 960, 32, "px",
    ),
    # ------------------------------------------------ grip recognition
    Tunable(
        "knob.engage_pinch", "Pinch tightness to grab",
        "How close thumb and index tip must be — as a fraction of hand size — "
        "to count as gripping the knob. Lower it if the knob engages when you "
        "don't mean to; raise it if grabbing feels unreliable.",
        "float", "Grip recognition", 0.20, 0.60, 0.01,
    ),
    Tunable(
        "knob.release_pinch", "Spread to release",
        "Open the pinch past this to let go. Kept a safe gap above the grab "
        "threshold (hysteresis) so a held grip can't flutter between grabbed "
        "and released.",
        "float", "Grip recognition", 0.45, 1.00, 0.01,
    ),
    Tunable(
        "knob.curl_reject_gap", "Relaxed-hand rejection",
        "Blocks a grab when the hand looks like a fist AND the middle "
        "fingertip sits within this distance of the thumb (as a fraction of "
        "hand size) — the signature of a hand resting in a curl, which "
        "otherwise reads as a pinch. To calibrate: watch 'curl gap' in Live "
        "tracking — relax your hand and note the number, then pinch "
        "deliberately and note it again; set this slider between the two. "
        "0 turns the check off.",
        "float", "Grip recognition", 0.0, 0.8, 0.01,
    ),
    Tunable(
        "knob.engage_frames", "Grab hold time",
        "Consecutive camera frames (~33 ms each) the pinch must hold, with the "
        "hand still, before the knob engages. Higher = fewer accidental grabs, "
        "but a moment slower to start.",
        "int", "Grip recognition", 1, 15, 1, "frames",
    ),
    Tunable(
        "knob.release_frames", "Release hold time",
        "Frames the pinch must stay open before the grip actually releases. "
        "Higher rides through single-frame tracking flickers mid-twist.",
        "int", "Grip recognition", 1, 15, 1, "frames",
    ),
    Tunable(
        "knob.max_engage_speed", "Max hand speed to grab",
        "A hand moving faster than this (in frame-widths per second) can't "
        "engage the knob — it blocks grabs while you're reaching for something "
        "or waving. Lower = grabs require a more deliberate, still hand.",
        "float", "Grip recognition", 0.1, 2.0, 0.05, "w/s",
    ),
    Tunable(
        "knob.hand_lost_grace_s", "Dropout grace",
        "If tracking loses your hand mid-grip for less than this, the grip "
        "survives and picks up where it left off.",
        "float", "Grip recognition", 0.0, 1.0, 0.05, "s",
    ),
    # ------------------------------------------------ turn feel & smoothing
    Tunable(
        "knob.deadband_deg", "Grab deadband",
        "Rotation ignored around the angle where you grabbed, so the act of "
        "gripping never nudges the volume. Raise it if the volume twitches "
        "when you grab or release.",
        "float", "Turn feel & smoothing", 0, 15, 0.5, "°",
    ),
    Tunable(
        "knob.filter_min_cutoff", "Steadiness at rest",
        "Smoothing floor (One Euro filter). Lower = the dial is rock steady "
        "while your hand hovers, at the cost of a touch of lag; higher = more "
        "immediate but can jitter with camera noise. Takes effect on the next "
        "grip.",
        "float", "Turn feel & smoothing", 0.1, 5.0, 0.1,
    ),
    Tunable(
        "knob.filter_beta", "Responsiveness in motion",
        "How aggressively smoothing relaxes when you twist quickly. Higher = "
        "fast twists track your hand more tightly. Takes effect on the next "
        "grip.",
        "float", "Turn feel & smoothing", 0.0, 0.10, 0.001,
    ),
    Tunable(
        "knob.max_frame_delta_deg", "Glitch rejection",
        "A single frame claiming more rotation than this is treated as a "
        "tracking glitch and ignored rather than applied to the volume.",
        "float", "Turn feel & smoothing", 10, 90, 5, "°",
    ),
    # ------------------------------------------------ who counts as a hand
    Tunable(
        "gate.min_score", "Detection confidence",
        "Hand detections the camera is less confident about are ignored "
        "entirely. Raise it to kill ghost detections (cushion patterns, "
        "faces); lower it if your real hand gets ignored in dim light.",
        "float", "Who counts as a hand", 0.30, 0.95, 0.05,
    ),
    Tunable(
        "gate.min_hand_frac", "Minimum hand size",
        "Hands smaller than this fraction of the frame height are ignored — "
        "filters out people far in the background.",
        "float", "Who counts as a hand", 0.02, 0.15, 0.005,
    ),
    Tunable(
        "gate.use_depth", "Distance gating",
        "Use the Kinect's depth stream to ignore hands outside the distance "
        "band below.",
        "bool", "Who counts as a hand",
    ),
    Tunable(
        "gate.depth_min_m", "Nearest distance",
        "Hands closer than this are ignored.",
        "float", "Who counts as a hand", 0.3, 2.0, 0.1, "m",
    ),
    Tunable(
        "gate.depth_max_m", "Farthest distance",
        "Hands farther than this are ignored — someone walking past behind "
        "the couch can't touch your volume.",
        "float", "Who counts as a hand", 1.0, 6.0, 0.1, "m",
    ),
    # ------------------------------------------------ swipe
    Tunable(
        "swipe.enabled", "Swipes enabled",
        "Swipe left/right to skip tracks.",
        "bool", "Swipe (track skip)",
    ),
    Tunable(
        "swipe.two_finger", "Two-finger swipe pose",
        "On: swipe with index+middle extended, other fingers curled — a "
        "deliberate pose that rarely happens by accident, so detection can be "
        "generous with fast motion. Off: the old open-palm swipe. The Live "
        "tracking 'pose' readout shows 'two' when the pose is being read "
        "correctly.",
        "bool", "Swipe (track skip)",
    ),
    Tunable(
        "swipe.invert", "Invert swipe direction",
        "Flip which way means next vs previous. If swiping right skips "
        "backwards for you, toggle this.",
        "bool", "Swipe (track skip)",
    ),
    Tunable(
        "swipe.min_presence_s", "Settle time",
        "A hand must have been in frame this long before it's allowed to "
        "swipe — so a hand entering the picture never skips a track.",
        "float", "Swipe (track skip)", 0.0, 2.0, 0.1, "s",
    ),
    Tunable(
        "swipe.min_travel_frac", "Required travel",
        "How far the palm must move sideways, as a fraction of the frame "
        "width. Higher = only big, committed swipes count.",
        "float", "Swipe (track skip)", 0.05, 0.5, 0.01,
    ),
    Tunable(
        "swipe.min_speed_frac", "Required speed",
        "Minimum swipe speed in frame-widths per second. Higher = lazy drifts "
        "of the hand won't skip.",
        "float", "Swipe (track skip)", 0.2, 3.0, 0.05, "w/s",
    ),
    Tunable(
        "swipe.max_vertical_ratio", "Horizontal strictness",
        "How horizontal the motion must be: the vertical component may be at "
        "most this multiple of the horizontal one. Lower = stricter — "
        "diagonal waves won't register.",
        "float", "Swipe (track skip)", 0.2, 2.0, 0.05,
    ),
    Tunable(
        "swipe.cooldown_s", "Cooldown",
        "Minimum time between swipes, so one gesture can't double-skip.",
        "float", "Swipe (track skip)", 0.2, 3.0, 0.1, "s",
    ),
    # ------------------------------------------------ fist hold
    Tunable(
        "fist.enabled", "Fist hold enabled",
        "Hold a closed fist still to toggle play/pause.",
        "bool", "Fist hold (play/pause)",
    ),
    Tunable(
        "fist.hold_s", "Hold duration",
        "How long the fist must be held still before play/pause fires.",
        "float", "Fist hold (play/pause)", 0.2, 3.0, 0.1, "s",
    ),
    Tunable(
        "fist.cooldown_s", "Cooldown",
        "Minimum time between fist-hold triggers.",
        "float", "Fist hold (play/pause)", 0.5, 5.0, 0.1, "s",
    ),
    Tunable(
        "fist.max_speed_frac", "Max speed while holding",
        "The fist must stay slower than this (frame-widths/s) for the whole "
        "hold — a moving fist is just a moving hand.",
        "float", "Fist hold (play/pause)", 0.05, 1.0, 0.05, "w/s",
    ),
)

_SPEC: dict[str, Tunable] = {t.key: t for t in TUNABLES}


class Tuning:
    """Applies tunable values onto the live AppConfig and persists overrides."""

    def __init__(self, cfg: AppConfig, path: str | None = None):
        self.cfg = cfg
        self.path = Path(path or os.environ.get("KK_TUNING_PATH", _DEFAULT_PATH))
        self._lock = threading.Lock()
        # Baseline = whatever defaults + yaml + env produced; "reset" returns
        # here, and only deltas from it are written to disk.
        self._base: dict[str, Any] = {t.key: self._get(t.key) for t in TUNABLES}
        self._load()

    # -- config access -------------------------------------------------
    def _get(self, key: str) -> Any:
        section, _, attr = key.partition(".")
        return getattr(getattr(self.cfg, section), attr)

    def _put(self, key: str, value: Any) -> None:
        section, _, attr = key.partition(".")
        setattr(getattr(self.cfg, section), attr, value)

    @staticmethod
    def _coerce(spec: Tunable, value: Any) -> Any:
        if spec.kind == "bool":
            return bool(value)
        v = float(value)
        v = min(max(v, spec.min), spec.max)
        return int(round(v)) if spec.kind == "int" else round(v, 4)

    def _apply_pair_rails(self, key: str, value: float) -> float:
        for lower, upper, gap in PAIRED:
            if key == lower:
                value = min(value, self._get(upper) - gap)
            elif key == upper:
                value = max(value, self._get(lower) + gap)
        return value

    # -- public API -----------------------------------------------------
    def set_value(self, key: str, value: Any) -> Any:
        """Validate, clamp, apply live, persist. Returns the value as applied
        (which may differ from the request after clamping/pair rails)."""
        spec = _SPEC.get(key)
        if spec is None:
            raise KeyError(key)
        with self._lock:
            v = self._coerce(spec, value)
            if spec.kind != "bool":
                v = self._apply_pair_rails(key, v)
                v = self._coerce(spec, v)  # rails could push past range ends
            self._put(key, v)
            self._save_locked()
            log.info("tuning: %s = %s", key, v)
            return v

    def reset(self) -> None:
        with self._lock:
            for key, value in self._base.items():
                self._put(key, value)
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove %s", self.path, exc_info=True)
            log.info("tuning: reset to baseline")

    def schema(self) -> list[dict]:
        """Full schema + live values for the dashboard, in display order."""
        out = []
        for t in TUNABLES:
            d = asdict(t)
            d["value"] = self._get(t.key)
            d["default"] = self._base[t.key]
            out.append(d)
        return out

    # -- persistence ----------------------------------------------------
    def _save_locked(self) -> None:
        deltas = {
            key: self._get(key)
            for key in _SPEC
            if self._get(key) != self._base[key]
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(deltas, indent=2, sort_keys=True))
            tmp.replace(self.path)
        except OSError:
            # Tuning still applies live; it just won't survive a restart.
            log.warning("could not persist tuning to %s", self.path, exc_info=True)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            log.warning("ignoring unreadable tuning file %s", self.path, exc_info=True)
            return
        applied = 0
        for key, value in data.items():
            spec = _SPEC.get(key)
            if spec is None:
                continue  # stale key from an older version — ignore
            self._put(key, self._coerce(spec, value))
            applied += 1
        if applied:
            log.info("applied %d saved tuning override(s) from %s", applied, self.path)
