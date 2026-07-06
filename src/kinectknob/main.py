"""Runtime wiring.

Thread layout (built for latency — every stage always works on the freshest
data and never queues stale frames):

  capture thread   -> reads the device, mirrors, drops into a latest-frame slot
  vision loop      -> (main thread) tracker + gesture engine on the newest frame
  asyncio thread   -> HA WebSocket client + controller + web server, one loop

If the capture device stalls (a known libfreenect2 long-run failure mode) the
process exits non-zero so the container restart policy power-cycles the stack.
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

from .capture import CaptureError, create_capture
from .config import AppConfig
from .controller import Controller
from .gestures.engine import GestureEngine
from .ha.client import HAClient
from .state import SharedState
from .types import Frame

log = logging.getLogger("kk.main")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

EXIT_CAPTURE_FAILURE = 3


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


class App:
    def __init__(self, cfg: AppConfig, preview: bool = False):
        self.cfg = cfg
        self.preview = preview
        self.stop_event = threading.Event()
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    def run(self) -> int:
        ensure_model(self.cfg.model_path)

        # Import here: mediapipe takes ~1s to import; keep --help snappy.
        from .tracking.hand_tracker import HandTracker

        try:
            capture = create_capture(self.cfg.capture)
            capture.start()
        except CaptureError as exc:
            log.critical("capture failed to start: %s", exc)
            return EXIT_CAPTURE_FAILURE
        self.shared.backend = capture.name
        self.shared.has_depth = capture.has_depth

        tracker = HandTracker(self.cfg.model_path, self.cfg.num_hands, self.cfg.mp_delegate)
        engine = GestureEngine(self.cfg)

        threads = [
            threading.Thread(target=self._capture_loop, args=(capture,), name="capture", daemon=True),
            threading.Thread(target=self._asyncio_thread, name="asyncio", daemon=True),
        ]
        for t in threads:
            t.start()

        self._install_signal_handlers()
        try:
            self._vision_loop(tracker, engine)
        finally:
            self.stop_event.set()
            capture.stop()
            tracker.close()
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass  # loop closed between the check and the call
            for t in threads:
                t.join(timeout=3)
        log.info("shut down (exit code %d)", self.exit_code)
        return self.exit_code

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("received signal %s — shutting down", signum)
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # ------------------------------------------------------------------
    def _capture_loop(self, capture) -> None:
        import cv2

        mirror = self.cfg.capture.mirror
        try:
            while not self.stop_event.is_set():
                frame = capture.read()
                if frame is None:
                    continue
                if mirror:
                    # cv2.flip is a plain SIMD copy — ~3x faster than
                    # materialising a negative-stride numpy view.
                    frame.rgb = cv2.flip(frame.rgb, 1)
                    if frame.depth_mm is not None:
                        frame.depth_mm = cv2.flip(frame.depth_mm, 1)
                self.slot.put(frame)
        except CaptureError as exc:
            log.critical("capture device failed: %s", exc)
            self.exit_code = EXIT_CAPTURE_FAILURE
            self.stop_event.set()
        except Exception:  # noqa: BLE001
            log.exception("capture thread crashed")
            self.exit_code = EXIT_CAPTURE_FAILURE
            self.stop_event.set()

    # ------------------------------------------------------------------
    def _vision_loop(self, tracker, engine: GestureEngine) -> None:
        import cv2

        proc_w = self.cfg.capture.proc_width
        fps_ema = 0.0
        proc_ema = 0.0
        last_t = time.monotonic()
        no_frame_since = time.monotonic()

        while not self.stop_event.is_set():
            frame = self.slot.get(timeout=0.5)
            if frame is None:
                if time.monotonic() - no_frame_since > 15.0:
                    log.critical("no frames for 15s — giving up")
                    self.exit_code = EXIT_CAPTURE_FAILURE
                    break
                continue
            no_frame_since = time.monotonic()

            t0 = time.monotonic()
            rgb = frame.rgb
            fh, fw = rgb.shape[:2]
            if fw > proc_w:
                scale = proc_w / fw
                rgb = cv2.resize(rgb, (proc_w, int(round(fh * scale))), interpolation=cv2.INTER_AREA)
            ph, pw = rgb.shape[:2]

            hands = tracker.process(rgb, frame.t)

            depth_sampler = None
            if frame.depth_mm is not None:
                depth_sampler = _make_depth_sampler(frame.depth_mm, pw, ph)

            events = engine.update(hands, frame.t, pw, ph, depth_sampler)
            if events:
                self.controller.submit(events)

            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
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

        web_app = create_app(self.cfg, self.shared, self.controller)
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
