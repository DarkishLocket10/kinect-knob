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
import threading
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
        self._exp_logged: Optional[tuple[float, float]] = None
        self._next_exp_log = 0.0
        self._depth_cache: Optional[np.ndarray] = None
        # "Proper photo" stacking (see capture_photo); guarded by _photo_lock.
        self._photo_lock = threading.Lock()
        self._photo_want = 0
        self._photo_acc: Optional[np.ndarray] = None
        self._photo_n = 0
        self._photo_result: Optional[np.ndarray] = None
        self._photo_done = threading.Event()
        # IR "proper photo" stacking (capture_ir_photo); same pattern, same lock.
        self._irphoto_want = 0
        self._irphoto_acc: Optional[np.ndarray] = None
        self._irphoto_n = 0
        self._irphoto_result: Optional[np.ndarray] = None
        self._irphoto_done = threading.Event()
        # Latest full-res color-aligned depth (mm) for region_depth(); the
        # ToF sensor self-illuminates, so this stays valid in a dark room.
        self._board_depth: Optional[np.ndarray] = None
        self._board_depth_t = 0.0
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
        self._apply_exposure()

    def _apply_exposure(self) -> None:
        """Send the configured color exposure to the device (must be running).
        Best-effort: a failure leaves the camera on its own auto-exposure."""
        spec = self.cfg.exposure
        if spec == "auto":
            return
        try:
            from .kv2_exposure import apply_exposure

            apply_exposure(self._device, spec)
            log.info("color exposure set to %r (watch the exposure/gain log "
                     "lines to confirm the sensor took it)", spec)
        except Exception:  # noqa: BLE001 — exposure is an enhancement, never fatal
            log.warning("could not set color exposure %r — camera stays on "
                        "auto", spec, exc_info=True)

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
        self._log_exposure(color, t)
        raw = color.to_array()                       # (1080, 1920, 4) uint8
        # libfreenect2 emits BGRX on Linux/libusb builds, but some pipelines
        # produce RGBX — trust the frame's own format over an assumption.
        fmt = getattr(getattr(color, "format", None), "name", "")
        self._photo_accumulate(raw, fmt)
        if ir is not None and self._irphoto_want:
            self._irphoto_accumulate(ir.to_array())

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
                # Registration is pure geometry — it works on dark color
                # frames — so the aligned board-depth cache stays fresh at
                # night too (region_depth serves the whiteboard reader's
                # obstruction check around the clock). ~1/s is plenty.
                if self._seq % 30 == 0 or self._board_depth is None:
                    self._update_board_depth(color, depth)
                return self._ir_frame(ir, depth, t)
        # Stash a full-res copy about once a second for the snapshot endpoint /
        # whiteboard reader (~6 MB convert+copy per second — negligible). Kept
        # unmirrored so scene text reads correctly.
        fullres = None
        if self._seq % 30 == 0:
            fullres = cv2.cvtColor(
                raw, cv2.COLOR_RGBA2BGR if fmt == "RGBX" else cv2.COLOR_BGRA2BGR
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
                self._board_depth = aligned.copy()   # full-res, for region_depth
                self._board_depth_t = time.time()
                depth_small = cv2.resize(aligned, size, interpolation=cv2.INTER_NEAREST)
                self._depth_cache = np.nan_to_num(
                    depth_small, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
            except Exception:  # noqa: BLE001 — depth alignment is best-effort,
                # and on failure the gate keeps using the previous cached map
                log.debug("registration failed for a frame", exc_info=True)

        self._seq += 1
        return Frame(rgb=rgb, depth_mm=self._depth_cache, t=t, seq=self._seq, fullres=fullres)

    def _log_exposure(self, color, t: float) -> None:
        """Surface the sensor's live exposure/gain (stamped on every color
        frame by the firmware) so exposure settings are field-verifiable from
        `docker logs`. Logs on material change, at most every 10 s."""
        if t < self._next_exp_log:
            return
        try:
            from freenect2 import lib

            exp = float(lib.freenect2_frame_get_exposure(color._c_object))
            gain = float(lib.freenect2_frame_get_gain(color._c_object))
        except Exception:  # noqa: BLE001 — telemetry only
            self._next_exp_log = t + 60.0
            return
        prev = self._exp_logged
        if prev is not None and abs(exp - prev[0]) < 0.15 * max(prev[0], 1e-3) \
                and abs(gain - prev[1]) < 0.15 * max(prev[1], 1e-3):
            return
        log.info("color sensor exposure %.2f ms, gain %.2f", exp, gain)
        self._exp_logged = (exp, gain)
        self._next_exp_log = t + 10.0

    # -- proper photos (multi-frame stacking) ----------------------------
    def capture_photo(self, frames: int = 8, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Take a "proper photo" for the whiteboard reader: average the next
        ``frames`` consecutive full-res color frames. The scene is static, so
        temporal stacking denoises like a long exposure (SNR ~ sqrt(N)) — and
        since this binding exposes no exposure control on the color camera,
        stacking is the strongest quality lever available. Blocks the CALLING
        thread (the web server's executor); the capture thread accumulates
        inside read(). Returns unmirrored BGR uint8, or None on timeout."""
        with self._photo_lock:
            self._photo_want = max(1, int(frames))
            self._photo_acc = None
            self._photo_n = 0
            self._photo_result = None
            self._photo_done.clear()
            done = self._photo_done
        if not done.wait(timeout):
            with self._photo_lock:
                self._photo_want = 0
                self._photo_acc = None
            return None
        return self._photo_result

    def _photo_accumulate(self, raw: np.ndarray, fmt: str) -> None:
        """Called by read() on every color frame; a cheap no-op unless a
        capture_photo() request is pending."""
        with self._photo_lock:
            if not self._photo_want:
                return
            bgr = cv2.cvtColor(
                raw, cv2.COLOR_RGBA2BGR if fmt == "RGBX" else cv2.COLOR_BGRA2BGR
            )
            if self._photo_acc is None or self._photo_acc.shape != bgr.shape:
                self._photo_acc = bgr.astype(np.float32)
                self._photo_n = 1
            else:
                self._photo_acc += bgr
                self._photo_n += 1
            if self._photo_n >= self._photo_want:
                mean = self._photo_acc / self._photo_n
                # Flip like the cached fullres: unmirrored, scene text readable.
                self._photo_result = cv2.flip(
                    np.clip(np.rint(mean), 0, 255).astype(np.uint8), 1)
                self._photo_want = 0
                self._photo_acc = None
                self._photo_done.set()

    def capture_ir_photo(self, frames: int = 8, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Average the next ``frames`` ACTIVE-IR frames and tone-map to uint8
        RGB. The ToF sensor self-illuminates, so this works in a pitch-black
        room. Requires ir_mode != off (otherwise IR frames never stream and
        this times out to None). Flipped like the color photo so scene text
        reads correctly."""
        with self._photo_lock:
            self._irphoto_want = max(1, int(frames))
            self._irphoto_acc = None
            self._irphoto_n = 0
            self._irphoto_result = None
            self._irphoto_done.clear()
            done = self._irphoto_done
        if not done.wait(timeout):
            with self._photo_lock:
                self._irphoto_want = 0
                self._irphoto_acc = None
            return None
        return self._irphoto_result

    def _irphoto_accumulate(self, arr: np.ndarray) -> None:
        """Called by read() when an IR photo request is pending."""
        with self._photo_lock:
            if not self._irphoto_want:
                return
            ir = np.asarray(arr, dtype=np.float32)
            if ir.ndim == 3:
                ir = ir[..., 0]
            if self._irphoto_acc is None or self._irphoto_acc.shape != ir.shape:
                self._irphoto_acc = ir.copy()
                self._irphoto_n = 1
            else:
                self._irphoto_acc += ir
                self._irphoto_n += 1
            if self._irphoto_n >= self._irphoto_want:
                mean = self._irphoto_acc / self._irphoto_n
                # Tone-map the STACKED float frame (cleaner than stacking
                # tone-mapped uint8), then flip to match the unmirrored color.
                self._irphoto_result = cv2.flip(ir_to_rgb(mean), 1)
                self._irphoto_want = 0
                self._irphoto_acc = None
                self._irphoto_done.set()

    def _update_board_depth(self, color, depth) -> None:
        """Refresh the full-res color-aligned depth cache (IR night mode —
        the color path refreshes it inside its own registration block)."""
        try:
            _, _, big_depth = self._device.registration.apply(
                color, depth, with_big_depth=True
            )
            big = big_depth.to_array()
            self._board_depth = big[1:-1, :].copy()
            self._board_depth_t = time.time()
        except Exception:  # noqa: BLE001 — stats endpoint just stays stale
            log.debug("board-depth registration failed", exc_info=True)

    def region_depth(self, x1: int, y1: int, x2: int, y2: int) -> Optional[dict]:
        """Depth stats (mm) over a region given in UNMIRRORED full-res color
        coordinates — the same frame /api/snapshot serves, so a whiteboard
        crop region can be passed verbatim. The aligned depth map follows the
        RAW (mirrored) color stream, hence the x flip. Self-illuminated ToF:
        valid day or pitch-black night. None until a depth frame has landed."""
        arr = self._board_depth
        if arr is None:
            return None
        h, w = arr.shape
        x1, x2 = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
        y1, y2 = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
        if x2 <= x1 or y2 <= y1:
            return None
        sub = arr[y1:y2, w - x2:w - x1]
        valid = sub[np.isfinite(sub) & (sub > 0)]
        out = {
            "age_s": round(time.time() - self._board_depth_t, 1),
            "valid_frac": round(float(valid.size) / sub.size, 3),
        }
        if valid.size:
            p = np.percentile(valid, [5, 10, 50, 90])
            out.update(p05_mm=int(p[0]), p10_mm=int(p[1]),
                       p50_mm=int(p[2]), p90_mm=int(p[3]))
        return out

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
