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
        # CAP_DSHOW avoids multi-second MSMF startup and exposure lag on Windows.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.cfg.webcam_index, backend)
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
