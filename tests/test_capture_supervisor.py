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


def test_repeated_failures_eventually_hand_over_to_the_container(monkeypatch):
    """In-process recovery is cheaper, but a device that keeps dying needs the
    USB stack power-cycled — which only a container restart does."""
    from kinectknob import main

    app = object.__new__(main.App)
    app._recycles = []
    app.exit_code = 0

    class _Shared:
        recycles = 0
    app.shared = _Shared()

    t = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: t[0])

    for i in range(main._MAX_RECYCLES):
        assert main.App._may_recycle(app) is True, f"recycle {i} should be allowed"
        t[0] += 1.0
    assert main.App._may_recycle(app) is False
    assert app.exit_code == main.EXIT_CAPTURE_FAILURE


def test_failures_spread_out_over_time_never_exhaust_the_budget(monkeypatch):
    """One stall every few hours is the documented normal for this host's USB
    controller. It must never accumulate into an exit."""
    from kinectknob import main

    app = object.__new__(main.App)
    app._recycles = []
    app.exit_code = 0

    class _Shared:
        recycles = 0
    app.shared = _Shared()

    t = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: t[0])
    for _ in range(50):
        assert main.App._may_recycle(app) is True
        t[0] += main._RECYCLE_WINDOW_S + 1.0
    assert app.exit_code == 0
