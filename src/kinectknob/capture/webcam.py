"""Plain webcam backend (OpenCV). No depth. Intended for development —
tune gestures on any laptop/PC before pointing the Kinect at the couch."""
from __future__ import annotations

import sys
import time
from typing import Optional

import cv2

from ..config import CaptureConfig
from ..types import Frame
from .base import CaptureBase, CaptureError


class WebcamCapture(CaptureBase):
    name = "webcam"
    has_depth = False

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0

    def start(self) -> None:
        cap = cv2.VideoCapture(self.cfg.webcam_index, self._api())
        if not cap.isOpened():
            raise CaptureError(
                f"could not open webcam index {self.cfg.webcam_index} — "
                "try another KK_WEBCAM_INDEX or check that no other app is using it"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read the freshest frame
        self._cap = cap

    def _api(self) -> int:
        """OpenCV capture API to open with.

        On Windows CAP_DSHOW historically avoided multi-second MSMF startup and
        exposure lag, so it stays the "auto" choice. That inverts on cameras
        behind the Intel Imaging Signal Processor (Surface Book/Pro), where
        DSHOW delivers ~16 fps and opens slower than MSMF's full ~30 — set
        KK_WEBCAM_API=msmf there.
        """
        name = (self.cfg.webcam_api or "auto").strip().lower()
        if name == "auto":
            return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        try:
            return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[name]
        except KeyError:
            raise CaptureError(
                f"unknown webcam api {name!r} — use auto, dshow, msmf or any"
            ) from None

    def read(self) -> Optional[Frame]:
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            time.sleep(0.05)
            return None
        self._seq += 1
        return Frame(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth_mm=None,
            t=time.monotonic(),
            seq=self._seq,
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
