"""Active-IR photo stacking and aligned-depth region stats (kinect2 backend).

No hardware needed: capture_ir_photo() blocks a caller thread while
_irphoto_accumulate() (normally driven by read()) feeds it frames, and
region_depth() reads a directly-injected aligned-depth array.
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


def _request_ir(cap, frames, timeout=5.0):
    box = {}
    th = threading.Thread(
        target=lambda: box.setdefault("img", cap.capture_ir_photo(frames, timeout)))
    th.start()
    for _ in range(200):
        with cap._photo_lock:
            if cap._irphoto_want:
                break
        time.sleep(0.005)
    return th, box


def _ir_raw(left: float, right: float) -> np.ndarray:
    """A 6x8 float32 IR frame: left half `left`, right half `right`."""
    ir = np.empty((6, 8), dtype=np.float32)
    ir[:, :4] = left
    ir[:, 4:] = right
    return ir


def test_ir_photo_is_tonemapped_mean_and_unmirrored():
    cap = _make_cap()
    th, box = _request_ir(cap, frames=2)
    cap._irphoto_accumulate(_ir_raw(0.0, 65535.0))
    cap._irphoto_accumulate(_ir_raw(0.0, 65535.0))
    th.join(5)
    img = box["img"]
    assert img is not None and img.shape == (6, 8, 3) and img.dtype == np.uint8
    # bright right half flipped to the LEFT; sqrt tone map: full scale -> 255
    assert int(img[0, 0, 0]) == 255
    assert int(img[0, 7, 0]) == 0


def test_ir_photo_accepts_hxwx1_frames_and_times_out_clean():
    cap = _make_cap()
    th, box = _request_ir(cap, frames=1)
    cap._irphoto_accumulate(_ir_raw(100.0, 100.0)[..., np.newaxis])
    th.join(5)
    assert box["img"].shape == (6, 8, 3)
    t0 = time.monotonic()
    assert cap.capture_ir_photo(frames=4, timeout=0.15) is None
    assert time.monotonic() - t0 < 2.0
    with cap._photo_lock:
        assert cap._irphoto_want == 0 and cap._irphoto_acc is None


def test_ir_accumulator_is_a_noop_without_a_request():
    cap = _make_cap()
    cap._irphoto_accumulate(_ir_raw(10.0, 10.0))
    with cap._photo_lock:
        assert cap._irphoto_n == 0 and cap._irphoto_acc is None


def test_region_depth_flips_x_and_reports_percentiles():
    cap = _make_cap()
    # aligned depth follows the MIRRORED color: a 10x20 map, LEFT half (in
    # mirrored coords) at 1000mm, RIGHT half at 3000mm
    arr = np.empty((10, 20), dtype=np.float32)
    arr[:, :10] = 1000.0
    arr[:, 10:] = 3000.0
    cap._board_depth = arr
    cap._board_depth_t = time.time()
    # unmirrored x 0..10 == mirrored x 10..20 -> the 3000mm half... after the
    # flip, unmirrored LEFT maps onto mirrored RIGHT
    out = cap.region_depth(0, 0, 10, 10)
    assert out["p50_mm"] == 3000 and out["valid_frac"] == 1.0
    out = cap.region_depth(10, 0, 20, 10)
    assert out["p50_mm"] == 1000


def test_region_depth_ignores_invalid_pixels_and_clamps():
    cap = _make_cap()
    arr = np.zeros((10, 20), dtype=np.float32)
    arr[:, :4] = 2500.0                       # only 20% of pixels valid
    arr[0, 0] = np.nan
    cap._board_depth = arr
    cap._board_depth_t = time.time()
    out = cap.region_depth(-5, -5, 999, 999)  # clamped to the full map
    assert out["p50_mm"] == 2500
    assert 0.19 <= out["valid_frac"] <= 0.21
    assert cap.region_depth(5, 5, 5, 9) is None       # empty region
    empty = _make_cap()
    assert empty.region_depth(0, 0, 5, 5) is None     # no depth yet
