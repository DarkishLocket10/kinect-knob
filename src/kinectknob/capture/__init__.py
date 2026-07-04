"""Capture backend factory with USB auto-detection.

Auto-detection scans Linux sysfs for Microsoft's Kinect USB product IDs:
  v2 sensor:  045e:02c4 / 045e:02d8   (045e:02d9 is only the adapter's hub)
  v1 camera:  045e:02ae (retail) / 045e:02bf (Kinect for Windows)
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import CaptureConfig
from .base import CaptureBase, CaptureError

log = logging.getLogger("kk.cap")

_V2_SENSOR_PIDS = {"02c4", "02d8"}
_V1_CAMERA_PIDS = {"02ae", "02bf"}


def _scan_usb_ids() -> set[str]:
    """Return the set of '<vid>:<pid>' strings present on the USB bus (Linux)."""
    found: set[str] = set()
    root = Path("/sys/bus/usb/devices")
    if not root.is_dir():
        return found
    for dev in root.iterdir():
        try:
            vid = (dev / "idVendor").read_text().strip().lower()
            pid = (dev / "idProduct").read_text().strip().lower()
            found.add(f"{vid}:{pid}")
        except OSError:
            continue
    return found


def detect_kinect() -> str:
    """Return 'kinect2', 'kinect1', or '' based on connected USB devices."""
    ids = _scan_usb_ids()
    if any(f"045e:{pid}" in ids for pid in _V2_SENSOR_PIDS):
        return "kinect2"
    if any(f"045e:{pid}" in ids for pid in _V1_CAMERA_PIDS):
        return "kinect1"
    return ""


def create_capture(cfg: CaptureConfig) -> CaptureBase:
    backend = cfg.backend.lower()
    if backend == "auto":
        detected = detect_kinect()
        if detected:
            log.info("auto-detected %s on the USB bus", detected)
            backend = detected
        else:
            log.warning("no Kinect found on the USB bus — falling back to webcam")
            backend = "webcam"

    if backend == "kinect1":
        from .kinect_v1 import KinectV1Capture
        return KinectV1Capture(cfg)
    if backend == "kinect2":
        from .kinect_v2 import KinectV2Capture
        return KinectV2Capture(cfg)
    if backend == "webcam":
        from .webcam import WebcamCapture
        return WebcamCapture(cfg)
    raise CaptureError(f"unknown capture backend: {cfg.backend!r}")


__all__ = ["CaptureBase", "CaptureError", "create_capture", "detect_kinect"]
