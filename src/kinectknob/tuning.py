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
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .config import AppConfig

log = logging.getLogger("kk.tune")

_DEFAULT_PATH = "/app/data/tuning.json"
BACKUPS_KEPT = 20


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
    ("presence.off_frac", "presence.on_frac", 0.002),
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
        "capture.crop", "Crop-in (zoom)",
        "Zoom the tracked view onto the centre of the frame: 1.0 = the full "
        "camera view, 2.0 = only the middle half. Crop in until you fill the "
        "picture — everything outside the crop (doorways, the TV, people "
        "walking past) is invisible to tracking, and your hand gets more of "
        "the model's attention, which helps at distance. The Live tracking "
        "view shows exactly what the cropped tracker sees, so drag this "
        "while watching it. Applies to the depth gate too (rgb and depth "
        "are cropped together).",
        "float", "Who counts as a hand", 1.0, 2.5, 0.05, "x",
    ),
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
    Tunable(
        "gate.object_gap_m", "Held-object rejection",
        "A hand holding something (bottle, mug, toothbrush) is 'busy': it "
        "can't grab the knob, swipe, or toggle playback — use your other "
        "hand, which takes over automatically when both are visible. "
        "Detected via the Kinect's depth: the held object's surface sits at "
        "least this much nearer the camera than the wrist. Lower = stricter "
        "rejection but more chance an empty hand reads as busy; 0 disables. "
        "Live values show as 'obj_gap' / 'holding' in Live tracking.",
        "float", "Who counts as a hand", 0.0, 0.30, 0.01, "m",
    ),
    Tunable(
        "gate.busy_linger_s", "Busy-hand linger",
        "How long a hand stays 'busy' after the held object was last seen — "
        "rides through depth flicker so a bottle in the hand can't briefly "
        "grab the knob between noisy frames.",
        "float", "Who counts as a hand", 0.0, 3.0, 0.1, "s",
    ),
    Tunable(
        "gate.lost_grace_s", "Dropout forgiveness",
        "Tracking losing the hand for less than this keeps its identity, "
        "presence and swipe motion history alive — a fast swipe blurs the "
        "hand enough to vanish for a frame or two, and without this grace "
        "every fast swipe died mid-gesture. Raise it if fast swipes still "
        "get eaten; 0 restores the old drop-everything behaviour.",
        "float", "Who counts as a hand", 0.0, 1.0, 0.05, "s",
    ),
    Tunable(
        "capture.low_light_boost", "Low-light boost",
        "Auto-brighten dim camera frames (gamma lift) before hand tracking. "
        "Keeps hands trackable in a dim room — especially with the shutter "
        "capped (KK_EXPOSURE) to prevent motion blur, which darkens the "
        "image. No effect on bright scenes or in IR night mode.",
        "bool", "Performance",
    ),
    # ------------------------------------------------ presence / idling
    Tunable(
        "presence.enabled", "Idle when the room is empty",
        "Pause hand tracking while nobody is in front of the camera, and wake "
        "it the moment somebody is. Hand landmarking is most of the work this "
        "app does and an empty room does not gesture. The camera keeps "
        "streaming either way — whiteboard-sync photographs the boards "
        "through it exactly when nobody is home — so this saves the GPU/CPU "
        "work, not the sensor.",
        "bool", "Presence",
    ),
    Tunable(
        "presence.on_frac", "Wake sensitivity",
        "How much of the view must be occupied by something that is not part "
        "of the empty room before presence turns ON, as a fraction of the "
        "pixels the depth camera can see. A person standing in frame reads "
        "tens of percent, so the default 0.02 wakes well before you are close "
        "enough to gesture. Raise it if a pet or a swinging door wakes the "
        "pipeline; the live reading is 'occupancy' in /api/presence.",
        "float", "Presence", 0.002, 0.20, 0.002,
    ),
    Tunable(
        "presence.off_frac", "Sleep sensitivity",
        "Occupancy must fall below this before the room counts as empty "
        "again. Kept under the wake threshold (hysteresis) so somebody "
        "sitting at the edge of the useful range can't flap the pipeline on "
        "and off.",
        "float", "Presence", 0.001, 0.15, 0.001,
    ),
    Tunable(
        "presence.linger_s", "Stay awake for",
        "How long presence survives after the last time somebody was seen. "
        "This is deliberately generous: stepping out of frame for a moment "
        "does not mean you have left, and it is also what whiteboard-sync "
        "reads to decide a board is safe to photograph.",
        "float", "Presence", 0.0, 120.0, 1.0, "s",
    ),
    Tunable(
        "presence.fg_gap_m", "Foreground depth gap",
        "How far in front of the learned empty scene something must sit "
        "before it counts as 'something new'. Lower sees more (and picks up "
        "depth noise on distant walls); higher only notices objects well "
        "clear of the background.",
        "float", "Presence", 0.05, 1.0, 0.05, "m",
    ),
    Tunable(
        "presence.far_m", "Presence range",
        "Ignore anything beyond this distance when judging presence — set it "
        "just past the far wall of the room you care about so the hallway "
        "behind you never counts as company.",
        "float", "Presence", 1.0, 8.0, 0.5, "m",
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
    # ------------------------------------------------ play/pause hold
    Tunable(
        "playpause.enabled", "Play/pause enabled",
        "Hold the pose still to toggle play/pause. The pose is an open palm "
        "facing the camera (or a closed fist via KK_PLAYPAUSE_POSE=fist).",
        "bool", "Play/pause hold",
    ),
    Tunable(
        "playpause.require_facing", "Palm must face the camera",
        "On: the open palm only counts when it actually faces the lens — the "
        "back of the hand or a sideways hand never toggles playback. Only "
        "applies to the palm pose.",
        "bool", "Play/pause hold",
    ),
    Tunable(
        "playpause.facing_min", "Facing strictness",
        "How squarely the palm must face the camera. A flat palm scores "
        "about 0.8, edge-on about 0. Raise this if playback still toggles "
        "on half-turned hands; lower it if you have to pose too precisely. "
        "The live score shows as 'facing' in /api/state while a hand is up.",
        "float", "Play/pause hold", 0.05, 0.7, 0.05, "",
    ),
    Tunable(
        "playpause.invert_facing", "Invert palm/back",
        "Flip this if the BACK of your hand triggers play/pause and your "
        "palm doesn't — the sign depends on the camera + MediaPipe build.",
        "bool", "Play/pause hold",
    ),
    Tunable(
        "playpause.spread_min", "Finger spread minimum",
        "Fingertips must be at least this far apart (relative to hand size). "
        "A hand wrapped around or pressed against a held object bunches its "
        "fingers (~0.15); a deliberate open palm spreads them (~0.4). "
        "0 disables. Live value shows as 'spread' in /api/state.",
        "float", "Play/pause hold", 0.0, 0.6, 0.05, "",
    ),
    Tunable(
        "playpause.object_gap_m", "Held-object depth gap",
        "Reject the hold when the palm surface is this much CLOSER to the "
        "camera than the wrist — something is being held in front of it. "
        "Needs the Kinect's depth. 0 disables.",
        "float", "Play/pause hold", 0.0, 0.3, 0.01, "m",
    ),
    Tunable(
        "playpause.hold_s", "Hold duration",
        "How long the pose must be held still before play/pause fires.",
        "float", "Play/pause hold", 0.2, 3.0, 0.1, "s",
    ),
    Tunable(
        "playpause.cooldown_s", "Cooldown",
        "Minimum time between play/pause triggers.",
        "float", "Play/pause hold", 0.5, 5.0, 0.1, "s",
    ),
    Tunable(
        "playpause.max_speed_frac", "Max speed while holding",
        "The hand must stay slower than this (frame-widths/s) for the whole "
        "hold — a hand in motion is just a hand in motion.",
        "float", "Play/pause hold", 0.05, 1.0, 0.05, "w/s",
    ),
    Tunable(
        "playpause.min_presence_s", "Min presence before hold",
        "The hand must have been in frame this long before a hold counts, so "
        "raising your hand into view never instantly toggles playback.",
        "float", "Play/pause hold", 0.0, 2.0, 0.05, "s",
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
        # Keys the user has explicitly set. Persisted with their values even
        # when they equal today's baseline: a hand-tuned value must never
        # drift because a code default or an env var changed underneath it.
        self._pinned: set[str] = set()
        # Keys in the file this version doesn't know (renamed/removed, or from
        # a newer version). Carried through every save untouched, so rolling
        # back a deploy never loses them.
        self._foreign: dict[str, Any] = {}
        # Saved values outside today's slider range: applied clamped, but the
        # file keeps the original until the user moves that slider.
        self._raw_kept: dict[str, Any] = {}
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
        if not math.isfinite(v):
            raise ValueError(f"{spec.key}: not a finite number")
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
            self._pinned.add(key)
            self._raw_kept.pop(key, None)
            self._save_locked()
            log.info("tuning: %s = %s", key, v)
            return v

    def reset_key(self, key: str) -> Any:
        """Return one setting to its baseline and stop pinning it."""
        if key not in _SPEC:
            raise KeyError(key)
        with self._lock:
            self._put(key, self._base[key])
            self._pinned.discard(key)
            self._raw_kept.pop(key, None)
            self._save_locked()
            log.info("tuning: %s reset to %s", key, self._base[key])
            return self._base[key]

    def reset(self) -> str | None:
        """Everything back to baseline. The current file is copied to a
        timestamped backup first (returned), so a reset can be undone with
        ``restore()`` — it used to delete hand-tuned values outright."""
        with self._lock:
            backup = self._backup_locked("reset")
            for key, value in self._base.items():
                self._put(key, value)
            self._pinned.clear()
            self._raw_kept.clear()
            try:
                if self._foreign:
                    self._save_locked()
                else:
                    self.path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove %s", self.path, exc_info=True)
            log.info("tuning: reset to baseline (backup: %s)", backup)
            return backup

    def export(self) -> dict[str, Any]:
        """The saved overrides, exactly as tuning.json holds them."""
        with self._lock:
            return self._deltas_locked()

    def import_values(self, values: dict[str, Any],
                      why: str = "import") -> tuple[dict, dict]:
        """Apply a dict of overrides through the normal validation. Returns
        (applied, rejected). Backs up the current file first."""
        with self._lock:
            self._backup_locked(why)
        applied, rejected = {}, {}
        for key, value in values.items():
            try:
                applied[key] = self.set_value(key, value)
            except KeyError:
                rejected[key] = "unknown setting"
            except (TypeError, ValueError) as exc:
                rejected[key] = str(exc)
        return applied, rejected

    def backups(self) -> list[str]:
        return sorted((p.name for p in self.path.parent.glob(
            self.path.stem + ".backup-*.json")), reverse=True)

    def restore(self, name: str | None = None) -> dict[str, Any]:
        """Re-apply a backup (default: the newest). Returns what was applied."""
        names = self.backups()
        if name is None:
            # The newest backup a person would mean: not the one a previous
            # restore took of itself (that would just flip-flop).
            usable = [n for n in names if not n.endswith("-restore.json")]
            if not usable:
                raise FileNotFoundError("no tuning backups")
            name = usable[0]
        if name not in names:   # only our own backup files, never arbitrary paths
            raise FileNotFoundError(name)
        data = json.loads((self.path.parent / name).read_text())
        applied, _ = self.import_values(data, why="restore")
        log.info("tuning: restored %d value(s) from %s", len(applied), name)
        return applied

    def _backup_locked(self, why: str) -> str | None:
        if not self.path.is_file():
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self.path.with_name(f"{self.path.stem}.backup-{stamp}-{why}.json")
        try:
            dest.write_bytes(self.path.read_bytes())
        except OSError:
            log.warning("could not back up %s", self.path, exc_info=True)
            return None
        for old in self.backups()[BACKUPS_KEPT:]:
            (self.path.parent / old).unlink(missing_ok=True)
        return dest.name

    def schema(self) -> list[dict]:
        """Full schema + live values for the dashboard, in display order."""
        out = []
        for t in TUNABLES:
            d = asdict(t)
            d["value"] = self._get(t.key)
            d["default"] = self._base[t.key]
            d["pinned"] = t.key in self._pinned
            out.append(d)
        return out

    # -- persistence ----------------------------------------------------
    def _deltas_locked(self) -> dict[str, Any]:
        out = dict(self._foreign)
        out.update({key: self._get(key) for key in _SPEC
                    if key in self._pinned or self._get(key) != self._base[key]})
        out.update(self._raw_kept)
        return out

    def _save_locked(self) -> None:
        deltas = self._deltas_locked()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_durable(self.path, json.dumps(deltas, indent=2, sort_keys=True))
        except OSError:
            # Tuning still applies live; it just won't survive a restart.
            log.warning("could not persist tuning to %s", self.path, exc_info=True)

    @property
    def _lastgood(self) -> Path:
        return self.path.with_name(self.path.stem + ".lastgood.json")

    def _read_file(self, path: Path) -> Optional[dict]:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _load(self) -> None:
        """Apply the saved overrides. A damaged file (a hard freeze mid-save,
        a bad hand edit) is moved aside — never overwritten — and the last
        copy that loaded cleanly is used instead. Before this, the next slider
        move rewrote the file with one key and every other value was gone."""
        if not self.path.is_file():
            return
        data = self._read_file(self.path)
        if data is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            bad = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}.json")
            try:
                self.path.replace(bad)
            except OSError:
                log.warning("could not move aside %s", self.path, exc_info=True)
            data = self._read_file(self._lastgood)
            log.error("tuning file %s was unreadable (kept as %s); %s", self.path,
                      bad.name, "using the last good copy" if data is not None
                      else "no last good copy — running on defaults")
            if data is None:
                return
            recovered = True
        else:
            recovered = False
        applied = 0
        for key, value in data.items():
            spec = _SPEC.get(key)
            if spec is None:
                # stale key from an older version (or one from a newer one):
                # not applied, but kept in the file — see _foreign
                self._foreign[key] = value
                continue
            try:
                v = self._coerce(spec, value)
            except (TypeError, ValueError):
                log.warning("tuning: ignoring bad saved value %s=%r", key, value)
                self._foreign[key] = value
                continue
            if spec.kind != "bool" and v != value:
                log.warning("tuning: saved %s=%r is outside today's range; "
                            "using %r (the file keeps %r)", key, value, v, value)
                self._raw_kept[key] = value   # never overwrite the hand-tuned value
            self._put(key, v)
            self._pinned.add(key)
            applied += 1
        if applied:
            log.info("applied %d saved tuning override(s) from %s", applied, self.path)
        with self._lock:
            if recovered:
                self._save_locked()
            try:
                if self.path.is_file():
                    _write_durable(self._lastgood, self.path.read_text())
            except OSError:
                log.debug("could not refresh %s", self._lastgood, exc_info=True)


def _write_durable(path: Path, text: str) -> None:
    """Write-aside, fsync, rename, fsync the directory: after a hard freeze
    the file is either the old version or the new one, never empty."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
