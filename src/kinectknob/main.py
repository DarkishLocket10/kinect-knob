"""Runtime wiring.

Thread layout (built for latency — every stage always works on the freshest
data and never queues stale frames):

  capture thread   -> reads the device, mirrors, drops into a latest-frame slot
  vision loop      -> (main thread) tracker + gesture engine on the newest frame
  asyncio thread   -> HA WebSocket client + controller + web server, one loop

Capture failure policy
----------------------
The asyncio thread (web server, HA client, controller) outlives the camera:
capture is acquired, supervised and recycled INSIDE ``run``, so a sensor that
vanishes or stalls no longer takes the process with it. Three cases, three
different answers:

* **Absent** — nothing on the USB bus. Wait in-process, forever, polling every
  few seconds. Restarting the container cannot conjure a USB device; doing it
  anyway is what produced 2204 restarts and a 61-second boot loop on
  2026-09-17, after the Kinect dropped off the bus and ``backend: auto`` fell
  back to a webcam this host does not have. The dashboard stays up and says
  what it is waiting for.
* **Stalled / failed to open** — present but misbehaving, the known
  libfreenect2 long-run failure mode. Re-open in-process, which is far cheaper
  than a container restart (no libfreenect2 rebuild, no OpenCL program build,
  no MediaPipe re-init, no lost tuning state).
* **Stalling repeatedly** — more than ``_MAX_RECYCLES`` recoveries inside
  ``_RECYCLE_WINDOW_S``. Now the USB stack itself is suspect, so exit non-zero
  and let the container restart policy power-cycle everything, as before.

Presence gating
---------------
Hand landmarking is the expensive half of the pipeline and it is pointless in
an empty room, so the vision loop only runs it while ``presence`` says someone
is actually there (see ``presence.py``). The capture device keeps streaming
either way — ``/api/snapshot`` serves whiteboard-sync precisely when nobody is
home — but the backend throttles depth registration while idle.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .capture import CaptureError, DeviceAbsent, create_capture
from .config import AppConfig
from .controller import Controller
from .gestures.engine import GestureEngine
from .ha.client import HAClient
from .presence import DepthPresence
from .state import SharedState
from .tuning import Tuning
from .types import Frame

log = logging.getLogger("kk.main")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

EXIT_CAPTURE_FAILURE = 3

_ABSENT_POLL_S = 5.0        # how often to re-scan the USB bus while waiting
_ABSENT_LOG_S = 300.0       # and how often to say so in the log
_MAX_RECYCLES = 4           # in-process capture recoveries before giving up...
_RECYCLE_WINDOW_S = 600.0   # ...counted over this rolling window
_NO_FRAME_TIMEOUT_S = 15.0  # vision loop sees nothing for this long -> recycle


class LatestFrameSlot:
    """Single-slot frame handoff: the vision loop always gets the newest frame,
    and frames that arrive while it's busy are silently replaced (dropped)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None

    def put(self, frame: Frame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self, timeout: float = 0.5) -> Optional[Frame]:
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    def clear(self) -> None:
        """Drop any frame left behind by a capture that is going away, so the
        next generation never hands the vision loop a dead device's frame."""
        with self._cond:
            self._frame = None


class App:
    def __init__(self, cfg: AppConfig, preview: bool = False):
        self.cfg = cfg
        self.preview = preview
        self.stop_event = threading.Event()      # process is shutting down
        self.cap_stop = threading.Event()        # THIS capture generation is done
        self.exit_code = 0
        self.slot = LatestFrameSlot()
        self.shared = SharedState()
        self.ha: Optional[HAClient] = None
        if cfg.ha.url and cfg.ha.token:
            entities = [cfg.ha.volume_entity, cfg.ha.media_entity]
            self.ha = HAClient(cfg.ha.url, cfg.ha.token, entities)
        else:
            log.warning("KK_HA_URL / KK_HA_TOKEN not set — running in dry-run mode")
        self.controller = Controller(cfg, self.ha)
        # Applies any saved dashboard-tuning overrides onto cfg immediately.
        self.tuning = Tuning(cfg)
        p = cfg.presence
        self.presence = DepthPresence(
            near_m=p.near_m, far_m=p.far_m, fg_gap_mm=p.fg_gap_m * 1000.0,
            on_frac=p.on_frac, off_frac=p.off_frac, linger_s=p.linger_s,
            warmup_s=p.warmup_s, static_absorb_s=p.static_absorb_s,
        )
        self.shared.presence = self.presence
        self._capture = None
        self._recycles: list[float] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    def run(self) -> int:
        ensure_model(self.cfg.model_path)

        # Import here: mediapipe takes ~1s to import; keep --help snappy.
        from .tracking.hand_tracker import HandTracker

        self._install_signal_handlers()
        # Web server / HA / controller come up FIRST and stay up across camera
        # generations: while the camera is missing, the dashboard is the only
        # thing that can tell you so.
        aio = threading.Thread(target=self._asyncio_thread, name="asyncio", daemon=True)
        aio.start()

        tracker = HandTracker(self.cfg.model_path, self.cfg.num_hands, self.cfg.mp_delegate)
        engine = GestureEngine(self.cfg)

        try:
            while not self.stop_event.is_set():
                capture = self._acquire_capture()
                if capture is None:
                    break
                self.cap_stop = threading.Event()
                cap_thread = threading.Thread(
                    target=self._capture_loop, args=(capture, self.cap_stop),
                    name="capture", daemon=True)
                cap_thread.start()
                try:
                    self._vision_loop(tracker, engine)
                finally:
                    self.cap_stop.set()
                    try:
                        capture.stop()
                    except Exception:  # noqa: BLE001 — teardown of a dead device
                        log.debug("capture.stop() raised during recycle", exc_info=True)
                    cap_thread.join(timeout=5)
                    self._release_capture()
                if self.stop_event.is_set():
                    break
                if not self._may_recycle():
                    break
        finally:
            self.stop_event.set()
            tracker.close()
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass  # loop closed between the check and the call
            aio.join(timeout=3)
        log.info("shut down (exit code %d)", self.exit_code)
        return self.exit_code

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("received signal %s — shutting down", signum)
            self.stop_event.set()
            self.cap_stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # -- capture supervision -------------------------------------------
    def _acquire_capture(self):
        """Open the camera, waiting as long as it takes. None = give up (the
        process is stopping, or the failure budget is spent)."""
        delay = 1.0
        absent_since: Optional[float] = None
        next_absent_log = 0.0
        while not self.stop_event.is_set():
            try:
                capture = create_capture(self.cfg.capture)
                capture.start()
            except DeviceAbsent as exc:
                now = time.monotonic()
                if absent_since is None:
                    absent_since = now
                    log.warning(
                        "%s — waiting in-process for it to enumerate; the web "
                        "server, Home Assistant link and last snapshot stay "
                        "available", exc)
                    next_absent_log = now + _ABSENT_LOG_S
                elif now >= next_absent_log:
                    log.warning("still no camera after %d min — waiting",
                                int((now - absent_since) // 60))
                    next_absent_log = now + _ABSENT_LOG_S
                self.shared.set_camera("absent", str(exc),
                                       waiting_s=now - absent_since)
                self.stop_event.wait(_ABSENT_POLL_S)
                continue
            except CaptureError as exc:
                log.error("capture failed to start: %s", exc)
                self.shared.set_camera("error", str(exc))
                if not self._may_recycle():
                    return None
                self.stop_event.wait(delay)
                delay = min(delay * 2, 30.0)
                continue
            except Exception:  # noqa: BLE001 — an unexpected backend blowup
                log.exception("capture failed to start unexpectedly")
                self.shared.set_camera("error", "unexpected backend failure")
                if not self._may_recycle():
                    return None
                self.stop_event.wait(delay)
                delay = min(delay * 2, 30.0)
                continue
            if absent_since is not None:
                log.info("camera came back after %.0f s of waiting",
                         time.monotonic() - absent_since)
            self._attach_capture(capture)
            return capture
        return None

    def _attach_capture(self, capture) -> None:
        self._capture = capture
        self.shared.backend = capture.name
        self.shared.has_depth = capture.has_depth
        # Kinect v2 can stack N consecutive frames into a "proper photo",
        # do the same on the active-IR stream, serve aligned-depth region
        # stats for the whiteboard reader's obstruction check, and answer
        # region OCCUPANCY (the sensitive per-region presence signal).
        self.shared.photo_fn = getattr(capture, "capture_photo", None)
        self.shared.ir_photo_fn = getattr(capture, "capture_ir_photo", None)
        self.shared.region_depth_fn = getattr(capture, "region_depth", None)
        self.shared.region_occupancy_fn = getattr(capture, "region_occupancy", None)
        self.shared.relearn_fn = getattr(capture, "relearn_regions", None)
        if hasattr(capture, "set_region_gap_mm"):
            capture.set_region_gap_mm(self.cfg.presence.region_gap_m * 1000.0,
                                      self.cfg.presence.region_static_absorb_s)
        self.shared.set_camera("ok", "")
        self.shared.generation += 1

    def _release_capture(self) -> None:
        self._capture = None
        self.slot.clear()
        self.shared.photo_fn = None
        self.shared.ir_photo_fn = None
        self.shared.region_depth_fn = None
        self.shared.region_occupancy_fn = None
        self.shared.relearn_fn = None
        # The background models describe a scene this camera generation saw;
        # a re-enumerated sensor can land with a different alignment.
        self.presence.force_relearn()

    def _may_recycle(self) -> bool:
        """Record a capture failure and decide whether to try again in-process.
        False means the budget is spent: exit non-zero so the container restart
        policy power-cycles the USB stack."""
        now = time.monotonic()
        self._recycles = [t for t in self._recycles if now - t < _RECYCLE_WINDOW_S]
        self._recycles.append(now)
        self.shared.recycles = len(self._recycles)
        if len(self._recycles) > _MAX_RECYCLES:
            log.critical(
                "camera failed %d times in %d min — exiting so the container "
                "restart policy power-cycles the USB stack",
                len(self._recycles), int(_RECYCLE_WINDOW_S // 60))
            self.exit_code = EXIT_CAPTURE_FAILURE
            return False
        log.warning("recycling the capture device in-process (%d/%d within %d min)",
                    len(self._recycles), _MAX_RECYCLES, int(_RECYCLE_WINDOW_S // 60))
        return True

    # ------------------------------------------------------------------
    def _capture_loop(self, capture, cap_stop: threading.Event) -> None:
        import cv2

        mirror = self.cfg.capture.mirror
        try:
            while not self.stop_event.is_set() and not cap_stop.is_set():
                frame = capture.read()
                if frame is None:
                    continue
                if mirror:
                    # cv2.flip is a plain SIMD copy — ~3x faster than
                    # materialising a negative-stride numpy view.
                    frame.rgb = cv2.flip(frame.rgb, 1)
                    if frame.depth_mm is not None:
                        frame.depth_mm = cv2.flip(frame.depth_mm, 1)
                if frame.fullres is not None:
                    # Deliberately NOT mirrored — see Frame.fullres.
                    self.shared.update_fullres(frame.fullres)
                self.slot.put(frame)
        except CaptureError as exc:
            log.critical("capture device failed: %s", exc)
            self.shared.set_camera("error", str(exc))
            cap_stop.set()
        except Exception:  # noqa: BLE001
            log.exception("capture thread crashed")
            self.shared.set_camera("error", "capture thread crashed")
            cap_stop.set()

    # ------------------------------------------------------------------
    def _vision_loop(self, tracker, engine: GestureEngine) -> None:
        """Runs until this capture generation ends (or the process stops)."""
        import cv2

        from .capture.crop import center_crop
        from .capture.lowlight import LowLightBoost

        booster = LowLightBoost()
        dt_ema = 0.0    # smooth the interval, then invert: EMA of 1/dt reads
        proc_ema = 0.0  # high when frame intervals alternate (Jensen bias)
        last_t = time.monotonic()
        no_frame_since = time.monotonic()
        cap_stop = self.cap_stop
        pcfg = self.cfg.presence
        next_probe = 0.0
        active = True
        last_dims = (self.cfg.capture.proc_width, self.cfg.capture.proc_width)
        nodepth_logged = False

        while not self.stop_event.is_set() and not cap_stop.is_set():
            frame = self.slot.get(timeout=0.5)
            if frame is None:
                if time.monotonic() - no_frame_since > _NO_FRAME_TIMEOUT_S:
                    log.critical("no frames for %.0fs — recycling the capture device",
                                 _NO_FRAME_TIMEOUT_S)
                    cap_stop.set()
                    break
                continue
            no_frame_since = time.monotonic()

            t0 = time.monotonic()
            # -- presence gate: skip the expensive half in an empty room ----
            # ``enabled`` is live-tunable, so this is re-read every frame.
            want_gate = pcfg.enabled and frame.depth_mm is not None
            if pcfg.enabled and frame.depth_mm is None and not nodepth_logged:
                log.info("presence gating needs a depth camera — this backend "
                         "has none, so the full pipeline stays on")
                nodepth_logged = True
            if want_gate:
                if t0 >= next_probe:
                    # Pick up any dashboard tuning before judging the frame.
                    self.presence.configure(pcfg)
                    res = self.presence.update(frame.depth_mm, t0)
                    next_probe = t0 + (pcfg.probe_s if res.present else pcfg.idle_probe_s)
                now_active = self.presence.present
            else:
                now_active = True
            if now_active != active:
                active = now_active
                log.info("presence: %s — %s hand tracking",
                         "someone is here" if active else "room is empty",
                         "resuming" if active else "pausing")
                if self._capture is not None:
                    self._capture.set_active(active)
                if not active:
                    # Hand the engine one empty frame so a knob left engaged
                    # releases cleanly instead of resuming minutes later.
                    pw, ph = last_dims
                    events = engine.update([], frame.t, pw, ph, None)
                    if events:
                        self.controller.submit(events)
            if not active:
                self.shared.update_idle(engine.snapshot())
                continue

            rgb = frame.rgb
            depth_mm = frame.depth_mm
            # Crop-in (live-tunable): zoom onto the centre so the user fills
            # the frame and motion at the edges (doorways, TV, passers-by)
            # never becomes a candidate hand. Depth MUST get the same crop —
            # the depth sampler maps tracked pixels by relative scale, which
            # only survives an equal-fraction crop of both arrays.
            zoom = self.cfg.capture.crop
            if zoom > 1.001:
                rgb = center_crop(rgb, zoom)
                if depth_mm is not None:
                    depth_mm = center_crop(depth_mm, zoom)
            fh, fw = rgb.shape[:2]
            proc_w = self.cfg.capture.proc_width  # live-tunable from the dashboard
            if fw > proc_w:
                scale = proc_w / fw
                rgb = cv2.resize(rgb, (proc_w, int(round(fh * scale))), interpolation=cv2.INTER_AREA)
            ph, pw = rgb.shape[:2]
            last_dims = (pw, ph)

            # Dim scenes (e.g. with the shutter capped against motion blur):
            # lift midtones so the tracker still sees the hand. IR frames are
            # already tone-mapped and self-illuminated — never boosted.
            if self.cfg.capture.low_light_boost and not frame.ir:
                rgb = booster.process(rgb)

            hands = tracker.process(rgb, frame.t)

            depth_sampler = None
            if depth_mm is not None:
                depth_sampler = _make_depth_sampler(depth_mm, pw, ph)

            events = engine.update(hands, frame.t, pw, ph, depth_sampler)
            if events:
                self.controller.submit(events)

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                dt_ema = 0.9 * dt_ema + 0.1 * dt if dt_ema else dt
            fps_ema = 1.0 / dt_ema if dt_ema else 0.0
            proc_ms = (now - t0) * 1000
            proc_ema = 0.9 * proc_ema + 0.1 * proc_ms if proc_ema else proc_ms

            self.shared.update_vision(rgb, hands, engine.snapshot(), fps_ema, proc_ema, ir=frame.ir)

            if self.preview:
                from . import debugdraw

                vol = self.controller.snapshot().get("volume")
                cv2.imshow("kinect-knob", debugdraw.render(rgb, hands, engine.snapshot(), vol))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop_event.set()

        if self.preview:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    def _asyncio_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self.controller.attach_loop(loop)

        import uvicorn

        from .web.server import create_app

        web_app = create_app(self.cfg, self.shared, self.controller, self.tuning)
        server = uvicorn.Server(
            uvicorn.Config(
                web_app,
                host=self.cfg.web.host,
                port=self.cfg.web.port,
                log_level="warning",
                loop="asyncio",
            )
        )

        def fail_fast(name: str):
            """A dead controller/web/HA task must stop the app loudly, not rot
            silently (e.g. web port already bound)."""
            def cb(task: asyncio.Task) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    log.critical("%s task died: %r", name, exc)
                    if self.exit_code == 0:
                        self.exit_code = 1
                    self.stop_event.set()
                    self.cap_stop.set()
            return cb

        named = [
            ("controller", loop.create_task(self.controller.run())),
            ("web-server", loop.create_task(server.serve())),
        ]
        if self.ha is not None:
            named.append(("ha-client", loop.create_task(self.ha.run())))
        for name, task in named:
            task.add_done_callback(fail_fast(name))
        tasks = [t for _, t in named]

        async def watch_stop():
            while not self.stop_event.is_set():
                await asyncio.sleep(0.2)
            server.should_exit = True
            for task in tasks:
                task.cancel()
            await asyncio.sleep(0.1)
            loop.stop()

        loop.create_task(watch_stop())
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:  # noqa: BLE001
                pass
            loop.close()


def _make_depth_sampler(depth_mm: np.ndarray, proc_w: int, proc_h: int):
    """Map proc-frame pixel coords -> metres via median of a 5x5 depth patch."""
    dh, dw = depth_mm.shape[:2]
    sx, sy = dw / proc_w, dh / proc_h

    def sample(x: float, y: float) -> Optional[float]:
        dx, dy = int(x * sx), int(y * sy)
        if not (0 <= dx < dw and 0 <= dy < dh):
            return None
        patch = depth_mm[max(0, dy - 2): dy + 3, max(0, dx - 2): dx + 3]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0

    return sample


def ensure_model(model_path: str) -> None:
    path = Path(model_path)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmark model (~7.5 MB) -> %s", path)
    tmp = path.with_suffix(".download")
    urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 — fixed https URL
    tmp.replace(path)


def run_app(cfg: AppConfig, preview: bool = False) -> int:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    return App(cfg, preview=preview).run()
