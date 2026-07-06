"""Regression for the 2026-07-06 memory leak: freenect2 0.2.3's Frame.create
allocates C++ frame memory with no destructor, so Registration.apply leaked
~10 MB per call (~144 MB/s live). Runs only where the binding is installed
(the Docker image); no Kinect hardware needed."""
import resource

import pytest

freenect2 = pytest.importorskip("freenect2")

from kinectknob.capture.kinect_v2 import _fix_frame_create_leak


def _peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_frame_create_is_gc_wrapped_and_does_not_leak():
    _fix_frame_create_leak()
    _fix_frame_create_leak()  # idempotent — second call must not double-wrap
    assert getattr(freenect2.Frame.create, "_kk_leakfix", False)

    # 200 big-depth-sized frames ≈ 1.6 GB if the C memory leaks. The pages must
    # be WRITTEN to count in RSS (in production, registration_apply fills
    # them), so fill each frame the way apply would. With the fix, each frame
    # is freed before the next allocates and peak RSS stays flat.
    before = _peak_rss_kb()
    for _ in range(200):
        frame = freenect2.Frame.create(1920, 1082, 4)
        frame.format = freenect2.FrameFormat.Float
        frame.to_array().fill(1.0)   # zero-copy view: touches all 8.3 MB
        del frame
    growth_mb = (_peak_rss_kb() - before) / 1024
    assert growth_mb < 200, f"Frame.create is leaking: peak RSS grew {growth_mb:.0f} MB"
