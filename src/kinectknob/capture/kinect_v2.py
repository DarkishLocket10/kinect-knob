"""Xbox One Kinect (v2) backend via the cffi-based ``freenect2`` package.

* Color: 1920x1080 @ 30fps, BGRX 4-channel on Linux. We downscale straight to
  the configured processing width (default 640) before anything else touches
  the pixels: resizing the 4-channel frame first and color-converting the
  small result is ~5x cheaper than converting at full res, and emitting
  proc-width frames means the vision loop never has to resize again.
* Depth: 512x424 time-of-flight, float32 mm. ``Registration.apply`` with
  ``with_big_depth=True`` yields a 1920x1082 depth map aligned to the color
  image (rows 1..1080 match color rows 0..1079); we crop and downscale it with
  nearest-neighbour so depth stays a per-pixel mm lookup in frame coordinates.
  Registration is the most expensive per-frame step and only feeds the
  distance gate, so it runs on every second frame and the frames in between
  reuse the cached map (a 3 m gate does not need 30 Hz depth).
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
from .kv2_stream import LatestQueueListener, read_latest

log = logging.getLogger("kk.cap.k2")

_FRAME_TIMEOUT_S = 5.0     # no frames for this long -> declare the device stalled
_DEPTH_EVERY_N = 2         # run registration on every Nth frame, cache between


def _fix_frame_create_leak() -> None:
    """freenect2 0.2.3's ``Frame.create()`` wraps ``freenect2_frame_create()``
    WITHOUT the ``ffi.gc`` destructor that device-streamed frames get, so every
    Python-allocated frame's C++ memory is permanent. ``Registration.apply``
    creates three such frames per call (undistorted + registered + big_depth ≈
    10 MB), which leaked ~144 MB/s on this box until the container OOM-threatened
    the host (2026-07-06). Re-point create() at a gc-wrapped allocation.
    Idempotent; the binding is pinned to 0.2.3."""
    import freenect2

    if getattr(freenect2.Frame.create, "_kk_leakfix", False):
        return
    ffi, lib = freenect2.ffi, freenect2.lib

    def create(cls, width, height, bytes_per_pixel):
        return cls(ffi.gc(
            lib.freenect2_frame_create(width, height, bytes_per_pixel),
            lib.freenect2_frame_dispose,
        ))

    create._kk_leakfix = True
    freenect2.Frame.create = classmethod(create)
    log.info("patched freenect2.Frame.create with gc-managed allocation (leak fix)")


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

        _fix_frame_create_leak()
        self._Device = Device
        self._FrameType = FrameType
        self._NoFrame = NoFrameReceivedError
        self._device = None
        self._running_ctx = None
        self._latest: dict = {}
        self._ir_switch = IrAutoSwitch(cfg.ir_mode)
        self._ir_active = self._ir_switch.active
        self._listener: LatestQueueListener | None = None
        self._drops_reported = 0
        self._next_drop_log = 0.0
        self._depth_cache: Optional[np.ndarray] = None
        # ir_mode is fixed for the process lifetime, so the needed-frames set is too.
        need_ir = cfg.ir_mode != "off"
        self._needed = frozenset(
            {FrameType.Color, FrameType.Depth} | ({FrameType.Ir} if need_ir else set())
        )

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
        # Replace the binding's default listener: its bare put_nowait raises
        # queue.Full inside the cffi C callback when the consumer lags, and
        # cffi then prints a traceback PER FRAME (~90/s) — a stderr/log storm
        # that can stagger the whole host. Ours drops the oldest frame
        # silently instead. All three attributes must be set: the two
        # properties rewire the C trampolines, and get_next_frame() reads
        # _default_listener (private attr — freenect2 is pinned to 0.2.3).
        self._listener = LatestQueueListener()
        self._device._default_listener = self._listener
        self._device.color_frame_listener = self._listener
        self._device.ir_and_depth_frame_listener = self._listener
        self._running_ctx = self._device.running()
        self._running_ctx.__enter__()
        log.info("Kinect v2 streaming (1080p color + ToF depth, GPU depth pipeline)")

    def read(self) -> Optional[Frame]:
        FrameType = self._FrameType
        need_ir = self._ir_switch.mode != "off"
        try:
            latest = read_latest(self._device, self._NoFrame, self._needed, _FRAME_TIMEOUT_S)
        except TimeoutError as exc:
            raise CaptureError(f"Kinect v2 stopped delivering frames — restarting ({exc})") from None
        color = latest[FrameType.Color]
        depth = latest[FrameType.Depth]
        ir = latest.get(FrameType.Ir)

        t = time.monotonic()
        # Dropping under load is by design (newest-wins), but make it visible
        # at a glance instead of silent: one log line per 10 s at most.
        dropped = self._listener.dropped if self._listener else 0
        if dropped > self._drops_reported and t >= self._next_drop_log:
            log.info(
                "capture running behind the sensor: %d frames dropped so far "
                "(newest-wins; expected under load)", dropped,
            )
            self._drops_reported = dropped
            self._next_drop_log = t + 10.0
        raw = color.to_array()                       # (1080, 1920, 4) uint8

        if need_ir:
            # Sparse-subsampled mean luma is plenty for the day/night decision.
            use_ir = self._ir_switch.update(float(raw[::16, ::16, :3].mean()))
            if use_ir != self._ir_active:
                self._ir_active = use_ir
                self._depth_cache = None  # depth res/alignment differs per mode
                log.info(
                    "%s (color luma %s the %s threshold for ~0.7 s)",
                    "IR night mode ON — tracking on active infrared" if use_ir
                    else "IR night mode OFF — back to color tracking",
                    "below" if use_ir else "above",
                    "dark" if use_ir else "bright",
                )
            if use_ir:
                return self._ir_frame(ir, depth, t)
        # Stash a full-res copy about once a second for the snapshot endpoint /
        # whiteboard reader (~6 MB convert+copy per second — negligible). Kept
        # unmirrored so scene text reads correctly.
        fullres = None
        if self._seq % 30 == 0:
            fmt_early = getattr(getattr(color, "format", None), "name", "")
            fullres = cv2.cvtColor(
                raw, cv2.COLOR_RGBA2BGR if fmt_early == "RGBX" else cv2.COLOR_BGRA2BGR
            )
            # libfreenect2 delivers the color stream horizontally MIRRORED
            # (verified against scene text). Flip to true orientation so
            # writing in the scene is readable — the whiteboard reader and
            # its left/right board semantics depend on this.
            fullres = cv2.flip(fullres, 1)

        # Downscale the 4-channel frame FIRST, then color-convert the small
        # result — same output, ~5x less pixel work than converting at 1080p.
        target_w = self.cfg.proc_width
        scale = target_w / raw.shape[1]
        size = (target_w, int(round(raw.shape[0] * scale)))
        small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        # libfreenect2 emits BGRX on Linux/libusb builds, but some pipelines
        # produce RGBX — trust the frame's own format over an assumption.
        fmt = getattr(getattr(color, "format", None), "name", "")
        if fmt == "RGBX":
            rgb = cv2.cvtColor(small, cv2.COLOR_RGBA2RGB)
        else:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGRA2RGB)

        if self._seq % _DEPTH_EVERY_N == 0 or self._depth_cache is None:
            try:
                _, _, big_depth = self._device.registration.apply(
                    color, depth, with_big_depth=True
                )
                big = big_depth.to_array()           # (1082, 1920) float32 mm
                aligned = big[1:-1, :]               # rows align with color
                depth_small = cv2.resize(aligned, size, interpolation=cv2.INTER_NEAREST)
                self._depth_cache = np.nan_to_num(
                    depth_small, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
            except Exception:  # noqa: BLE001 — depth alignment is best-effort,
                # and on failure the gate keeps using the previous cached map
                log.debug("registration failed for a frame", exc_info=True)

        self._seq += 1
        return Frame(rgb=rgb, depth_mm=self._depth_cache, t=t, seq=self._seq, fullres=fullres)

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
