"""Xbox One Kinect (v2) backend via the cffi-based ``freenect2`` package.

* Color: 1920x1080 @ 30fps, BGRX 4-channel on Linux. We downscale to
  ~960px width before handing frames on — plenty for hand landmarks and it
  keeps CPU flat.
* Depth: 512x424 time-of-flight, float32 mm. ``Registration.apply`` with
  ``with_big_depth=True`` yields a 1920x1082 depth map aligned to the color
  image (rows 1..1080 match color rows 0..1079); we crop and downscale it with
  nearest-neighbour so depth stays a per-pixel mm lookup in frame coordinates.
* IR: 512x424 active infrared off the same ToF sensor — self-illuminated, so
  it sees hands in a pitch-black room. When the color image goes dark
  (``ir_mode: auto``) we hand MediaPipe the tone-mapped IR image instead, and
  since IR and depth share the sensor the depth map is pixel-aligned with no
  registration pass.

Depth packet processing runs on the GPU: the Docker image builds libfreenect2
with OpenCL and sets LIBFREENECT2_PIPELINE=cl, which on a GTX 1080 Ti decodes
a depth frame in about a millisecond.

Known libfreenect2 failure mode: long-running sessions can hit USB bulk
transfer stalls (upstream issues #546/#547/#915) where frames just stop.
We surface that as a CaptureError after a timeout so the process exits and
the container restart policy brings the device back cleanly.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from ..config import CaptureConfig
from ..types import Frame
from .base import CaptureBase, CaptureError
from .ir import IrAutoSwitch, ir_to_rgb

log = logging.getLogger("kk.cap.k2")

_FRAME_TIMEOUT_S = 5.0     # no frames for this long -> declare the device stalled
_TARGET_WIDTH = 960        # downscale 1920 -> 960 for tracking


class KinectV2Capture(CaptureBase):
    name = "kinect2"
    has_depth = True

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self._seq = 0
        try:
            from freenect2 import Device, FrameType  # noqa: F401
        except ImportError as exc:
            raise CaptureError(
                "the 'freenect2' python module is not installed — run inside the "
                "kinect-knob Docker image, or use --backend webcam for development"
            ) from exc
        from freenect2 import Device, FrameType, NoFrameReceivedError

        self._Device = Device
        self._FrameType = FrameType
        self._NoFrame = NoFrameReceivedError
        self._device = None
        self._running_ctx = None
        self._latest: dict = {}
        self._ir_switch = IrAutoSwitch(cfg.ir_mode)
        self._ir_active = self._ir_switch.active

    def start(self) -> None:
        try:
            self._device = self._Device()
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(
                "Kinect v2 not found. Check: Kinect Adapter for Windows powered, "
                "plugged into a USB 3.0 port (Intel/Renesas controller — ASMedia "
                "does not work), and /dev/bus/usb passed into the container. "
                f"({exc})"
            ) from exc
        self._running_ctx = self._device.running()
        self._running_ctx.__enter__()
        log.info("Kinect v2 streaming (1080p color + ToF depth, GPU depth pipeline)")

    def read(self) -> Optional[Frame]:
        FrameType = self._FrameType
        deadline = time.monotonic() + _FRAME_TIMEOUT_S
        # Drain until we hold a fresh color+depth (and IR, unless disabled) set.
        need_ir = self._ir_switch.mode != "off"
        color = depth = ir = None
        while True:
            try:
                type_, frame = self._device.get_next_frame(timeout=1.0)
            except self._NoFrame:
                if time.monotonic() > deadline:
                    raise CaptureError(
                        "Kinect v2 stopped delivering frames (USB stall) — restarting"
                    ) from None
                continue
            if type_ is FrameType.Color:
                color = frame
            elif type_ is FrameType.Depth:
                depth = frame
            elif type_ is FrameType.Ir:
                ir = frame
            if color is not None and depth is not None and (ir is not None or not need_ir):
                break
            if time.monotonic() > deadline:
                raise CaptureError("Kinect v2 frame pairing timed out — restarting") from None

        t = time.monotonic()
        raw = color.to_array()                       # (1080, 1920, 4) uint8

        if need_ir:
            # Sparse-subsampled mean luma is plenty for the day/night decision.
            use_ir = self._ir_switch.update(float(raw[::16, ::16, :3].mean()))
            if use_ir != self._ir_active:
                self._ir_active = use_ir
                log.info(
                    "%s (color luma %s the %s threshold for ~0.7 s)",
                    "IR night mode ON — tracking on active infrared" if use_ir
                    else "IR night mode OFF — back to color tracking",
                    "below" if use_ir else "above",
                    "dark" if use_ir else "bright",
                )
            if use_ir:
                return self._ir_frame(ir, depth, t)
        # libfreenect2 emits BGRX on Linux/libusb builds, but some pipelines
        # produce RGBX — trust the frame's own format over an assumption.
        fmt = getattr(getattr(color, "format", None), "name", "")
        if fmt == "RGBX":
            rgb_full = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)
        else:
            rgb_full = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

        scale = _TARGET_WIDTH / rgb_full.shape[1]
        size = (_TARGET_WIDTH, int(round(rgb_full.shape[0] * scale)))
        rgb = cv2.resize(rgb_full, size, interpolation=cv2.INTER_AREA)

        depth_mm = None
        try:
            _, _, big_depth = self._device.registration.apply(
                color, depth, with_big_depth=True
            )
            big = big_depth.to_array()               # (1082, 1920) float32 mm
            aligned = big[1:-1, :]                   # rows align with color
            depth_small = cv2.resize(aligned, size, interpolation=cv2.INTER_NEAREST)
            depth_small = np.nan_to_num(depth_small, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = depth_small
        except Exception:  # noqa: BLE001 — depth alignment is best-effort
            log.debug("registration failed for a frame", exc_info=True)

        self._seq += 1
        return Frame(rgb=rgb, depth_mm=depth_mm, t=t, seq=self._seq)

    def _ir_frame(self, ir, depth, t: float) -> Frame:
        """Night mode: track on the tone-mapped IR image. IR and depth come off
        the same sensor, so the raw depth array is already pixel-aligned —
        no registration, and full native 512x424 resolution for both."""
        rgb = ir_to_rgb(ir.to_array())
        depth_mm = np.squeeze(depth.to_array()).astype(np.float32, copy=False)
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        self._seq += 1
        return Frame(rgb=rgb, depth_mm=depth_mm, t=t, seq=self._seq, ir=True)

    def stop(self) -> None:
        if self._running_ctx is not None:
            try:
                self._running_ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._running_ctx = None
        self._device = None
