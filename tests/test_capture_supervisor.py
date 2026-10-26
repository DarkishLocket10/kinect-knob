"""Camera acquisition policy — the 2026-09-17 boot loop.

The Kinect dropped off the USB bus; ``backend: auto`` fell back to a webcam
this host does not have; ``create_capture`` raised; the process exited; the
container restart policy started it again 61 seconds later; repeat 2204 times.
Each cycle paid a full libfreenect2 + OpenCL + MediaPipe startup to discover
the same absent device.

The fix is a distinction: a camera that is ABSENT is waited for, a camera that
is BROKEN is recycled, and only a camera that keeps breaking is worth a
container restart.
"""
from __future__ import annotations

import pytest

from kinectknob.capture import CaptureError, DeviceAbsent, create_capture
from kinectknob.config import CaptureConfig


@pytest.fixture
def no_kinect(monkeypatch):
    monkeypatch.setattr("kinectknob.capture.detect_kinect", lambda: "")


def test_absent_kinect_is_not_a_webcam_cue(no_kinect):
    """The boot loop itself: nothing on the bus must NOT silently become
    "try the webcam", because failing to open one exits the process."""
    with pytest.raises(DeviceAbsent) as exc:
        create_capture(CaptureConfig(backend="auto"))
    assert "USB bus" in str(exc.value)


def test_device_absent_is_distinguishable_from_a_broken_device(no_kinect):
    """Callers branch on the type, so DeviceAbsent must stay a CaptureError
    (old handlers keep working) while being separately catchable."""
    assert issubclass(DeviceAbsent, CaptureError)
    with pytest.raises(CaptureError):
        create_capture(CaptureConfig(backend="auto"))


def test_webcam_fallback_is_available_when_asked_for(no_kinect, monkeypatch):
    made = {}

    class FakeWebcam:
        name = "webcam"
        has_depth = False

        def __init__(self, cfg):
            made["cfg"] = cfg

    monkeypatch.setattr("kinectknob.capture.webcam.WebcamCapture", FakeWebcam)
    cap = create_capture(CaptureConfig(backend="auto", auto_fallback=True))
    assert cap.name == "webcam"
    assert made["cfg"].auto_fallback is True


def test_auto_prefers_a_detected_kinect(monkeypatch):
    monkeypatch.setattr("kinectknob.capture.detect_kinect", lambda: "kinect1")

    class FakeV1:
        name = "kinect1"
        has_depth = True

        def __init__(self, cfg):
            pass

    monkeypatch.setattr("kinectknob.capture.kinect_v1.KinectV1Capture", FakeV1)
    assert create_capture(CaptureConfig(backend="auto")).name == "kinect1"


def test_an_explicit_unknown_backend_still_fails_loudly():
    with pytest.raises(CaptureError):
        create_capture(CaptureConfig(backend="potato"))


def test_capture_failures_end_the_process(monkeypatch, tmp_path):
    """No in-process reopen: the binding's process-wide context hands a
    reopen the same stale device (2026-10-26), so a broken camera — as
    opposed to an absent one — must exit for a fresh process."""
    from kinectknob import main

    app = object.__new__(main.App)
    app.exit_code = 0
    app._exit_log = tmp_path / "exits.json"
    app.stop_event = __import__("threading").Event()
    app._stop_reason = None

    class _Shared:
        def set_camera(self, *a, **k):
            pass
    app.shared = _Shared()

    class _Window:
        def active(self):
            return False
    app.off_window = _Window()

    def boom(cfg):
        raise CaptureError("Kinect v2 not found")

    monkeypatch.setattr(main, "create_capture", boom)
    monkeypatch.setattr(main.App, "_request_replug", lambda self: None)
    assert main.App._acquire_capture(app) is None
    assert app.exit_code == main.EXIT_CAPTURE_FAILURE
