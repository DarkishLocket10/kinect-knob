"""Configuration: defaults -> optional YAML file -> environment overrides.

Environment variables (all prefixed ``KK_``) win over the YAML file so secrets
and per-host settings stay out of config files in Docker deployments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class HAConfig:
    url: str = ""                       # e.g. http://192.168.1.10:8123
    token: str = ""                     # long-lived access token
    volume_entity: str = ""             # e.g. media_player.bose_soundbar_700
    media_entity: str = ""              # defaults to volume_entity if empty
    max_volume: float = 1.0             # safety ceiling for gesture-set volume
    volume_step: float = 0.02           # fallback relative step when no state known
    send_interval_s: float = 0.10       # min gap between volume_set calls while twisting


@dataclass
class CaptureConfig:
    backend: str = "auto"               # auto | kinect1 | kinect2 | webcam
    webcam_index: int = 0
    width: int = 640                    # webcam request; kinect backends fix their own
    height: int = 480
    fps: int = 30
    mirror: bool = True                 # selfie view: your right = image right
    proc_width: int = 640               # frames wider than this are downscaled before tracking
    ir_mode: str = "auto"               # Kinect v2 night mode: auto | off | always


@dataclass
class GateConfig:
    use_depth: bool = True              # only honoured when the backend has depth
    depth_min_m: float = 0.5
    depth_max_m: float = 3.0
    min_hand_frac: float = 0.045        # min hand size as fraction of frame height
    min_score: float = 0.55             # ignore low-confidence (ghost) detections


@dataclass
class KnobConfig:
    engage_pinch: float = 0.42          # pinch ratio below this begins engagement
    release_pinch: float = 0.65         # pinch ratio above this releases (hysteresis)
    engage_frames: int = 4              # ~130 ms of held pinch before engaging
    curl_reject_gap: float = 0.42       # fist pose + middle tip this close to thumb = resting curl (0 = off)
    release_frames: int = 5
    full_scale_deg: float = 270.0       # degrees of rotation for 0% -> 100% volume
    deadband_deg: float = 3.0           # ignore this much wobble around the grab point
    invert: bool = False                # flip rotation direction
    filter_min_cutoff: float = 1.2      # One Euro baseline cutoff (Hz)
    filter_beta: float = 0.015          # One Euro speed coefficient
    max_frame_delta_deg: float = 40.0   # reject implausible per-frame jumps
    hand_lost_grace_s: float = 0.30     # keep the knob gripped through brief dropouts
    max_engage_speed: float = 0.6       # don't engage while palm moves faster (widths/s)


@dataclass
class SwipeConfig:
    enabled: bool = True
    two_finger: bool = True             # swipe pose: index+middle out (else open palm)
    invert: bool = False                # flip which direction means next/previous
    window_s: float = 0.35              # motion window evaluated for a swipe
    min_travel_frac: float = 0.18       # min horizontal travel (fraction of frame width)
    min_speed_frac: float = 0.80        # min mean speed (frame-widths per second)
    max_vertical_ratio: float = 0.60    # |dy| must be < this * |dx|
    cooldown_s: float = 0.8
    min_presence_s: float = 0.35        # hand must be visible this long before a swipe


@dataclass
class FistConfig:
    enabled: bool = False               # play/pause on fist-hold (opt-in)
    hold_s: float = 0.7
    max_speed_frac: float = 0.25
    cooldown_s: float = 2.0


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8420
    debug_stream: bool = True           # MJPEG overlay stream (renders only while watched)


@dataclass
class AppConfig:
    ha: HAConfig = field(default_factory=HAConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    knob: KnobConfig = field(default_factory=KnobConfig)
    swipe: SwipeConfig = field(default_factory=SwipeConfig)
    fist: FistConfig = field(default_factory=FistConfig)
    web: WebConfig = field(default_factory=WebConfig)
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = 2
    mp_delegate: str = "cpu"            # hand landmarker inference: cpu | gpu
    log_level: str = "INFO"


# env var -> (section, field, type). type "bool" accepts 1/true/yes/on.
_ENV_MAP: dict[str, tuple[str, str, str]] = {
    "KK_HA_URL": ("ha", "url", "str"),
    "KK_HA_TOKEN": ("ha", "token", "str"),
    "KK_VOLUME_ENTITY": ("ha", "volume_entity", "str"),
    "KK_MEDIA_ENTITY": ("ha", "media_entity", "str"),
    "KK_MAX_VOLUME": ("ha", "max_volume", "float"),
    "KK_VOLUME_STEP": ("ha", "volume_step", "float"),
    "KK_BACKEND": ("capture", "backend", "str"),
    "KK_WEBCAM_INDEX": ("capture", "webcam_index", "int"),
    "KK_MIRROR": ("capture", "mirror", "bool"),
    "KK_PROC_WIDTH": ("capture", "proc_width", "int"),
    "KK_IR_MODE": ("capture", "ir_mode", "str"),
    "KK_USE_DEPTH": ("gate", "use_depth", "bool"),
    "KK_DEPTH_MIN_M": ("gate", "depth_min_m", "float"),
    "KK_DEPTH_MAX_M": ("gate", "depth_max_m", "float"),
    "KK_MIN_HAND_FRAC": ("gate", "min_hand_frac", "float"),
    "KK_FULL_SCALE_DEG": ("knob", "full_scale_deg", "float"),
    "KK_INVERT_ROTATION": ("knob", "invert", "bool"),
    "KK_DEADBAND_DEG": ("knob", "deadband_deg", "float"),
    "KK_SWIPE_ENABLED": ("swipe", "enabled", "bool"),
    "KK_SWIPE_TWO_FINGER": ("swipe", "two_finger", "bool"),
    "KK_INVERT_SWIPE": ("swipe", "invert", "bool"),
    "KK_PLAYPAUSE_ENABLED": ("fist", "enabled", "bool"),
    "KK_PORT": ("web", "port", "int"),
    "KK_DEBUG_STREAM": ("web", "debug_stream", "bool"),
    "KK_MODEL_PATH": ("", "model_path", "str"),
    "KK_NUM_HANDS": ("", "num_hands", "int"),
    "KK_MP_DELEGATE": ("", "mp_delegate", "str"),
    "KK_LOG_LEVEL": ("", "log_level", "str"),
}

_TRUE = {"1", "true", "yes", "on"}


def _coerce(raw: str, kind: str) -> Any:
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return raw.strip().lower() in _TRUE
    return raw


def _apply_dict(cfg: AppConfig, data: dict) -> None:
    for section_name, section_val in data.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section: {section_name!r}")
        target = getattr(cfg, section_name)
        if isinstance(section_val, dict):
            for key, val in section_val.items():
                if not hasattr(target, key):
                    raise ValueError(f"Unknown config key: {section_name}.{key}")
                cur = getattr(target, key)
                if isinstance(cur, bool) and not isinstance(val, bool):
                    val = str(val).strip().lower() in _TRUE
                elif isinstance(cur, float):
                    val = float(val)
                elif isinstance(cur, int) and not isinstance(val, bool):
                    val = int(val)
                setattr(target, key, val)
        else:
            setattr(cfg, section_name, section_val)


def load_config(path: Optional[str] = None) -> AppConfig:
    cfg = AppConfig()

    yaml_path = path or os.environ.get("KK_CONFIG", "")
    if yaml_path and Path(yaml_path).is_file():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

    for env, (section, key, kind) in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None or raw == "":
            continue
        value = _coerce(raw, kind)
        target = cfg if section == "" else getattr(cfg, section)
        setattr(target, key, value)

    cfg.capture.ir_mode = cfg.capture.ir_mode.strip().lower()
    if cfg.capture.ir_mode not in ("auto", "off", "always"):
        raise ValueError(f"capture.ir_mode must be auto|off|always, got {cfg.capture.ir_mode!r}")
    cfg.mp_delegate = cfg.mp_delegate.strip().lower()
    if cfg.mp_delegate not in ("cpu", "gpu"):
        raise ValueError(f"mp_delegate must be cpu|gpu, got {cfg.mp_delegate!r}")
    if not cfg.ha.media_entity:
        cfg.ha.media_entity = cfg.ha.volume_entity
    cfg.ha.max_volume = min(max(cfg.ha.max_volume, 0.0), 1.0)
    # A zero/negative step would make the relative-detent loop spin forever.
    cfg.ha.volume_step = min(max(cfg.ha.volume_step, 0.001), 0.5)
    return cfg
