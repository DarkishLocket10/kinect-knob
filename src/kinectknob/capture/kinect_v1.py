"""Xbox 360 Kinect (v1) backend via libfreenect's sync Python wrapper.

* RGB: 640x480 @ 30fps, returned in RGB order by freenect (no conversion).
* Depth: DEPTH_REGISTERED = millimetres, already aligned to the RGB camera —
  a depth_mm[y, x] lookup matches rgb[y, x] directly.

freenect.sync_get_* return None when the device is missing; we treat a run of
failures as fatal so the container's restart policy can recover the device.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import CaptureConfig
from ..types import Frame
from .base import CaptureBase, CaptureError

log = logging.getLogger("kk.cap.k1")


class KinectV1Capture(CaptureBase):
    name = "kinect1"
    has_depth = True

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._seq = 0
        self._fail_count = 0
        try:
            import freenect  # noqa: F401 — installed by the Docker image build
        except ImportError as exc:
            raise CaptureError(
                "the 'freenect' python module is not installed — run inside the "
                "kinect-knob Docker image, or use --backend webcam for development"
            ) from exc
        self._freenect = __import__("freenect")

    def start(self) -> None:
        fn = self._freenect
        probe = fn.sync_get_video(0, fn.VIDEO_RGB)
        if probe is None:
            raise CaptureError(
                "Kinect v1 not responding. Check: 12V power adapter connected, "
                "USB plugged into the server, and /dev/bus/usb passed into the container."
            )
        log.info("Kinect v1 streaming (640x480 RGB + registered depth)")

    def read(self) -> Optional[Frame]:
        fn = self._freenect
        video = fn.sync_get_video(0, fn.VIDEO_RGB)
        if video is None:
            self._fail_count += 1
            if self._fail_count > 90:  # ~3s of nothing
                raise CaptureError("Kinect v1 stopped delivering frames (USB stall/unplug?)")
            time.sleep(0.03)
            return None
        self._fail_count = 0
        rgb = video[0]  # (480, 640, 3) uint8 RGB

        depth_mm = None
        d = fn.sync_get_depth(0, fn.DEPTH_REGISTERED)
        if d is not None:
            depth_mm = d[0]  # (480, 640) uint16 millimetres, RGB-aligned

        self._seq += 1
        return Frame(rgb=rgb, depth_mm=depth_mm, t=time.monotonic(), seq=self._seq)

    def stop(self) -> None:
        try:
            self._freenect.sync_stop()
        except Exception:  # noqa: BLE001
            pass
