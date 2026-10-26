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
* **Stalled / failed to open / wedged** — present but misbehaving. Close the
  device cleanly and EXIT; the container restarts a fresh process. There is no
  in-process reopen any more (2026-10-26): the pinned freenect2 binding keeps
  one process-wide context that it enumerates once and never forgets a device
  in, so a reopen was handed the same half-dead handle ("device @4:5 is
  already be open!") and failed within seconds — and disposing that context
  to force a fresh one deadlocks or segfaults against libfreenect2's
  packet-processor threads, both observed on the live sensor. A fresh process
  is the only teardown the kernel guarantees.
* **Failing repeatedly** — exits are remembered in ``data/capture_exits.json``;
  from the second inside ``_EXIT_MEMORY_S`` the next start rests first (30 s,
  doubling, 10 min cap), and from the third the sensor is logically unplugged
  and replugged (``usb_power.replug``) — the one reset that has recovered a
  sensor whose every open fails ("all submissions failed").

Scheduled off window
--------------------
``schedule.off_window`` (``KK_OFF_WINDOW``, e.g. ``20:00-06:30`` in
``KK_TZ``): the camera is closed and the sensor logically unplugged for the
window; the web server stays up and reports ``camera: "off"``. When the
window ends the sensor is plugged back in and the process exits 0 so the
container restarts it fresh — a clean daily restart of libusb, OpenCL and
MediaPipe, so nothing compounds across days. ``POST /api/camera/restart``
does the same restart (close, replug, exit 0) on demand.

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
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from .capture import CaptureError, DeviceAbsent, DeviceWedged, create_capture
from .capture import usb_power
from .config import AppConfig
from .controller import Controller
from .gestures.engine import GestureEngine
from .ha.client import HAClient
from .presence import DepthPresence
from .schedule import OffWindow
from .state import SharedState
from .tuning import Tuning
from .types import Frame

log = logging.getLogger("kk.main")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

EXIT_CAPTURE_FAILURE = 3
EXIT_CLEAN_RESTART = 0       # off window over / restart requested: not a failure

_ABSENT_POLL_S = 5.0        # how often to re-scan the USB bus while waiting
_ABSENT_LOG_S = 300.0       # and how often to say so in the log
_NO_FRAME_TIMEOUT_S = 15.0  # vision loop sees nothing for this long -> fail
_SCHEDULE_POLL_S = 15.0
# Capture-failure exits are remembered across restarts so a sensor that keeps
# failing backs off instead of restarting — and flashing — every few seconds.
_EXIT_LOG = Path("/app/data/capture_exits.json")
_EXIT_MEMORY_S = 1800.0
_REPLUG_AFTER = 3            # this many failure exits in memory -> replug too
# Replugs happen at the START of the next process, when nothing holds the
# device open — yanking it from under a live libusb handle is asking for the
# same native crashes that ruled out in-process teardown.
_REPLUG_MARKER = Path("/app/data/replug_on_start")
# stop/close normally takes well under a second. Native teardown that hangs
# (a libfreenect2 thread join, a wedged libusb call) must not leave the
# process alive and blind: hard-exit and let the container restart it.
_TEARDOWN_TIMEOUT_S = 20.0


def _rest_before_open_s(recent_failure_exits: int) -> float:
    """One failure is the documented daytime normal (a stall every hour or
    so): restart straight away. Repeats back off 30 s, doubling, max 10 min."""
    if recent_failure_exits < 2:
        return 0.0
    return min(30.0 * 2 ** (recent_failure_exits - 2), 600.0)


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
        self.off_window = OffWindow(cfg.schedule.off_window, cfg.schedule.tz)
        self.shared.schedule_fn = self.off_window.describe
        self.shared.restart_fn = self.request_restart
        # Why the current capture generation was ended on purpose (None =
        # it failed): "schedule" | "requested". Set before cap_stop.
        self._stop_reason: Optional[str] = None
        self._exit_log = _EXIT_LOG
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
        if self.off_window.enabled:
            log.info("camera off window %s (%s); next change %s",
                     self.off_window.spec, self.off_window.tz_name,
                     self.off_window.describe()["next_change"])
            threading.Thread(target=self._schedule_watch, name="schedule",
                             daemon=True).start()

        if not self.off_window.active():
            if self._take_replug_request():
                usb_power.replug()        # ends authorized
            else:
                # A process stopped mid-window (or killed) may have left the
                # sensor logically unplugged; outside the window it is on.
                usb_power.set_authorized(True)
        self._cool_down_after_failures()

        try:
            while not self.stop_event.is_set():
                if self.off_window.active():
                    self._sleep_through_off_window()
                    if not self.stop_event.is_set():
                        log.info("off window over — plugging the sensor back "
                                 "in and exiting for a clean scheduled restart")
                        usb_power.set_authorized(True)
                        # A night unplugged is the longest rest there is:
                        # yesterday's failures say nothing about this morning.
                        self._exit_log.unlink(missing_ok=True)
                        self.exit_code = EXIT_CLEAN_RESTART
                    break
                capture = self._acquire_capture()
                if capture is None:
                    if self._take_stop_reason() == "schedule":
                        continue
                    break
                self._stop_reason = None
                self.cap_stop = threading.Event()
                cap_thread = threading.Thread(
                    target=self._capture_loop, args=(capture, self.cap_stop),
                    name="capture", daemon=True)
                cap_thread.start()
                try:
                    self._vision_loop(tracker, engine)
                finally:
                    self.cap_stop.set()
                    self._release_knob(engine)
                    # Join BEFORE closing: the capture thread must be out of
                    # read() before the device is closed. A live stream hands
                    # it a frame within ~33 ms; a stalled one times out
                    # inside read() in <= 5 s.
                    cap_thread.join(timeout=7)
                    self._stop_capture(capture)
                    self._release_capture()
                if self.stop_event.is_set():
                    break
                reason = self._take_stop_reason()
                if reason == "schedule":
                    # The next process starts inside the window, never opens
                    # the device, and logically unplugs it for the night.
                    log.info("off window started — camera stopped; restarting "
                             "the process to switch the sensor off")
                    self.exit_code = EXIT_CLEAN_RESTART
                    break
                if reason == "requested":
                    log.info("camera restart requested — restarting the "
                             "process; the sensor is replugged on start")
                    self._request_replug()
                    self.exit_code = EXIT_CLEAN_RESTART
                    break
                self._fail_exit("camera failed")
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

    def _stop_capture(self, capture) -> None:
        watchdog = threading.Timer(_TEARDOWN_TIMEOUT_S, self._teardown_hung)
        watchdog.daemon = True
        watchdog.start()
        try:
            capture.stop()
        except Exception:  # noqa: BLE001 — teardown of a dead device
            log.debug("capture.stop() raised during recycle", exc_info=True)
        finally:
            watchdog.cancel()

    def _teardown_hung(self) -> None:
        log.critical("camera teardown hung for %.0f s — hard-exiting so the "
                     "container restarts the process (sensor replugged on start)",
                     _TEARDOWN_TIMEOUT_S)
        self._record_failure_exit()
        self._request_replug()
        logging.shutdown()
        os._exit(EXIT_CAPTURE_FAILURE)

    def _release_knob(self, engine: GestureEngine) -> None:
        """The camera went away mid-gesture: hand the engine an empty frame so
        an engaged knob releases now instead of resuming on the next camera."""
        try:
            w = self.cfg.capture.proc_width
            events = engine.update([], time.monotonic(), w, w, None)
            if events:
                self.controller.submit(events)
        except Exception:  # noqa: BLE001
            log.debug("knob release on camera stop raised", exc_info=True)

    # -- failure exits across restarts -------------------------------------
    def _recent_failure_exits(self) -> list[float]:
        try:
            data = json.loads(self._exit_log.read_text())
            now = time.time()
            return [t for t in data.get("exits", []) if now - t < _EXIT_MEMORY_S]
        except (OSError, ValueError, AttributeError):
            return []

    def _record_failure_exit(self) -> None:
        self.exit_code = EXIT_CAPTURE_FAILURE
        try:
            exits = self._recent_failure_exits() + [time.time()]
            self._exit_log.write_text(json.dumps({"exits": exits}))
        except OSError:
            log.debug("could not record capture failure exit", exc_info=True)

    def _fail_exit(self, why: str, force_replug: bool = False) -> None:
        """Every capture failure ends the process (see the module docstring).
        Record it; replug the sensor too when failures are piling up."""
        self._record_failure_exit()
        n = len(self._recent_failure_exits())
        log.critical("%s — exiting so the container restarts a fresh process "
                     "(%d failure exit(s) in the last %d min)", why, n,
                     int(_EXIT_MEMORY_S // 60))
        if force_replug or n >= _REPLUG_AFTER:
            self._request_replug()

    def _request_replug(self) -> None:
        try:
            _REPLUG_MARKER.touch()
        except OSError:
            log.debug("could not write the replug marker", exc_info=True)

    def _take_replug_request(self) -> bool:
        try:
            _REPLUG_MARKER.unlink()
            return True
        except OSError:
            return False

    def _cool_down_after_failures(self) -> None:
        n = len(self._recent_failure_exits())
        self.shared.recycles = n   # dashboard: failure restarts, last 30 min
        delay = _rest_before_open_s(n)
        if not delay:
            return
        log.warning("%d capture-failure restart(s) in the last %d min — letting "
                    "the sensor rest %.0f s before opening it", n,
                    int(_EXIT_MEMORY_S // 60), delay)
        self.shared.set_camera("error", f"recovering — resting {delay:.0f}s "
                                        f"after {n} failed restart(s)")
        until = time.monotonic() + delay
        while not self.off_window.active():   # the window is a longer rest
            left = until - time.monotonic()
            if left <= 0 or self.stop_event.wait(min(left, 5.0)):
                break

    # -- schedule / on-demand restart ------------------------------------
    def _take_stop_reason(self) -> Optional[str]:
        reason, self._stop_reason = self._stop_reason, None
        return reason

    def _end_generation(self, reason: str) -> None:
        self._stop_reason = reason
        self.cap_stop.set()

    def request_restart(self) -> bool:
        """Dashboard/API: close the camera, replug the sensor and restart the
        process — the same clean restart as the end of the off window; not
        counted as a failure. Refused inside the off window."""
        if self.off_window.active():
            return False
        log.info("camera restart requested")
        self._end_generation("requested")
        return True

    def _schedule_watch(self) -> None:
        """End the running capture generation when the off window opens."""
        while not self.stop_event.wait(_SCHEDULE_POLL_S):
            # Only a RUNNING camera needs ending; every other state notices
            # the window on its own (acquire, the rest, the loop top).
            if (self.off_window.active() and self._capture is not None
                    and self._stop_reason != "schedule"):
                log.info("off window %s started — stopping the camera",
                         self.off_window.spec)
                self._end_generation("schedule")

    def _sleep_through_off_window(self) -> None:
        until = self.off_window.describe()["next_change"]
        log.info("camera is OFF until %s (%s) — Kinect stream stopped and "
                 "device closed", until, self.off_window.tz_name)
        self.shared.set_camera("off", f"scheduled off until {until}")
        self.shared.clear_frames()
        # Closed is not off: un-configure the sensor for the night (the
        # kernel's view of an unplug — no streams, emitters dark).
        usb_power.set_authorized(False)
        while self.off_window.active() and not self.stop_event.wait(_SCHEDULE_POLL_S):
            pass

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
        """Open the camera, waiting as long as it takes for an absent one.
        None = the process should stop (shutting down, the off window opened,
        or the open failed — failures always exit, see the module docstring)."""
        absent_since: Optional[float] = None
        next_absent_log = 0.0
        while not self.stop_event.is_set():
            if self.off_window.active():
                self._stop_reason = "schedule"
                return None
            capture = None
            try:
                capture = create_capture(self.cfg.capture)
                capture.start()
            except DeviceWedged as exc:
                self.shared.set_camera("error", str(exc))
                # Keep the half-started device (and the frame listeners its
                # stuck colour transfers still call into) alive until exit.
                self._wedged_capture = capture
                # Replug: the reset that actually recovers this state
                # (libfreenect2's own reset-on-open does not), and it makes
                # the kernel cancel the stuck URBs.
                self._fail_exit(str(exc), force_replug=True)
                return None
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
                # Not retried in-process: the binding's context enumerated the
                # bus once and will keep failing the same way.
                self.shared.set_camera("error", str(exc))
                self._fail_exit(f"capture failed to start: {exc}")
                return None
            except Exception:  # noqa: BLE001 — an unexpected backend blowup
                log.exception("capture failed to start unexpectedly")
                self.shared.set_camera("error", "unexpected backend failure")
                self._fail_exit("capture failed to start unexpectedly")
                return None
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
        # Never serve a closed camera's last frame as a current photo.
        self.shared.clear_frames()
        self.shared.photo_fn = None
        self.shared.ir_photo_fn = None
        self.shared.region_depth_fn = None
        self.shared.region_occupancy_fn = None
        self.shared.relearn_fn = None

    # ------------------------------------------------------------------
    def _capture_loop(self, capture, cap_stop: threading.Event) -> None:
        import cv2

        mirror = self.cfg.capture.mirror
        try:
            while not self.stop_event.is_set() and not cap_stop.is_set():
                frame = capture.read()
                if frame is None:
                    continue
                if cap_stop.is_set():
                    break   # a stopped generation must never publish over the next
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
            if cap_stop.is_set():
                return   # torn down underneath us on purpose
            log.critical("capture device failed: %s", exc)
            self.shared.set_camera("error", str(exc))
            cap_stop.set()
        except Exception:  # noqa: BLE001
            if cap_stop.is_set():
                return
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
    code = App(cfg, preview=preview).run()
    # Skip interpreter teardown: it garbage-collects the freenect2 Device and
    # its frame listeners while libfreenect2's native threads may still call
    # into them (segfault, exit 139). The kernel reclaims the device anyway.
    logging.shutdown()
    sys.stdout.flush()
    os._exit(code)
