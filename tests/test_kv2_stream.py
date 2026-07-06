"""Kinect v2 stream plumbing: drop-oldest listener and newest-wins drain."""
from __future__ import annotations

import pytest

from kinectknob.capture.kv2_stream import LatestQueueListener, read_latest


class NoFrame(Exception):
    pass


class FakeDevice:
    """Scripted device: yields (type, frame) tuples, then raises NoFrame."""

    def __init__(self, frames):
        self._frames = list(frames)

    def get_next_frame(self, timeout=None):
        if not self._frames:
            raise NoFrame()
        return self._frames.pop(0)


class TestLatestQueueListener:
    def test_passthrough_below_capacity(self):
        li = LatestQueueListener(maxsize=4)
        li("color", 1)
        li("depth", 2)
        assert li.get(0) == ("color", 1)
        assert li.get(0) == ("depth", 2)
        assert li.dropped == 0

    def test_overflow_drops_oldest_never_raises(self):
        li = LatestQueueListener(maxsize=3)
        for i in range(10):          # 7 past capacity — the C callback path
            li("color", i)           # must survive all of them silently
        assert li.dropped == 7
        # Survivors are the newest three, in order.
        got = [li.get(0)[1] for _ in range(3)]
        assert got == [7, 8, 9]

    def test_interface_matches_binding(self):
        # freenect2's Device duck-types the listener: callable + .get(timeout)
        li = LatestQueueListener()
        assert callable(li)
        li("ir", object())
        assert li.get(0)[0] == "ir"


class TestReadLatest:
    def test_returns_newest_of_each_type(self):
        dev = FakeDevice([
            ("color", "c1"), ("depth", "d1"), ("color", "c2"),
            ("ir", "i1"), ("depth", "d2"),
        ])
        out = read_latest(dev, NoFrame, frozenset({"color", "depth", "ir"}), 5.0)
        assert out == {"color": "c2", "depth": "d2", "ir": "i1"}

    def test_backlog_is_fully_swept_not_fifo_popped(self):
        # 12 stale frames queued + fresh ones at the end: the fresh ones win.
        stale = [("color", f"old{i}") for i in range(12)]
        dev = FakeDevice(stale + [("color", "new"), ("depth", "d")])
        out = read_latest(dev, NoFrame, frozenset({"color", "depth"}), 5.0)
        assert out["color"] == "new"

    def test_ignores_unneeded_types(self):
        dev = FakeDevice([("ir", "i"), ("color", "c"), ("depth", "d")])
        out = read_latest(dev, NoFrame, frozenset({"color", "depth"}), 5.0)
        assert out["color"] == "c" and out["depth"] == "d"
        assert out.get("ir") == "i"  # kept, harmless

    def test_timeout_when_stream_missing(self):
        # Depth never arrives; a fake clock advances 2s per call.
        ticks = iter(range(0, 100, 2))
        dev = FakeDevice([("color", "c")])
        with pytest.raises(TimeoutError, match="depth"):
            read_latest(
                dev, NoFrame, frozenset({"color", "depth"}), 3.0,
                clock=lambda: float(next(ticks)),
            )

    def test_timeout_when_no_frames_at_all(self):
        ticks = iter(range(0, 100, 2))
        dev = FakeDevice([])
        with pytest.raises(TimeoutError, match="no frames"):
            read_latest(
                dev, NoFrame, frozenset({"color"}), 3.0,
                clock=lambda: float(next(ticks)),
            )
