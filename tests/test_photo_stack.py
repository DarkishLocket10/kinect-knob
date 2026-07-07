"""Multi-frame "proper photo" stacking on the Kinect v2 backend.

No hardware needed: capture_photo() blocks a caller thread while
_photo_accumulate() (normally driven by read()) feeds it frames.
"""
import threading
import time

import numpy as np
import pytest

pytest.importorskip("freenect2")

from kinectknob.capture.kinect_v2 import KinectV2Capture
from kinectknob.config import CaptureConfig


def _make_cap() -> KinectV2Capture:
    return KinectV2Capture(CaptureConfig())


def _request(cap, frames, timeout=5.0):
    """Arm a capture_photo request on a worker thread; returns (thread, box)."""
    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault("img", cap.capture_photo(frames, timeout)))
    th.start()
    for _ in range(200):                      # wait until the request is armed
        with cap._photo_lock:
            if cap._photo_want:
                break
        time.sleep(0.005)
    return th, box


def _raw(left: int, right: int) -> np.ndarray:
    """A 6x8 BGRX frame: left half `left`, right half `right`."""
    raw = np.empty((6, 8, 4), dtype=np.uint8)
    raw[:, :4] = left
    raw[:, 4:] = right
    return raw


def test_photo_is_the_mean_of_n_frames_and_unmirrored():
    cap = _make_cap()
    th, box = _request(cap, frames=3)
    for v in (10, 20, 40):                    # mean 23.33 -> 23
        cap._photo_accumulate(_raw(v, 100), "BGRX")
    th.join(5)
    img = box["img"]
    assert img is not None and img.shape == (6, 8, 3) and img.dtype == np.uint8
    # flipped horizontally: the bright right half is now on the LEFT
    assert int(img[0, 0, 0]) == 100
    assert int(img[0, 7, 0]) == 23


def test_photo_times_out_to_none_when_no_frames_arrive():
    cap = _make_cap()
    t0 = time.monotonic()
    assert cap.capture_photo(frames=4, timeout=0.15) is None
    assert time.monotonic() - t0 < 2.0
    with cap._photo_lock:                     # request cleaned up
        assert cap._photo_want == 0 and cap._photo_acc is None


def test_accumulator_is_a_noop_without_a_pending_request():
    cap = _make_cap()
    cap._photo_accumulate(_raw(50, 50), "BGRX")
    with cap._photo_lock:
        assert cap._photo_n == 0 and cap._photo_acc is None


def test_resolution_change_mid_stack_restarts_cleanly():
    cap = _make_cap()
    th, box = _request(cap, frames=2)
    cap._photo_accumulate(np.full((4, 4, 4), 200, np.uint8), "BGRX")
    cap._photo_accumulate(_raw(60, 60), "BGRX")   # shape change: restart
    cap._photo_accumulate(_raw(60, 60), "BGRX")
    th.join(5)
    assert box["img"].shape == (6, 8, 3)
    assert int(box["img"][0, 0, 0]) == 60
