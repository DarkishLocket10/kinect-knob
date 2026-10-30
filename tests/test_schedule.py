"""Scheduled camera-off window + idle throttle config (wall-pad node).

The Surface wall pad tracked hands on the CPU 24/7 (~2.3 cores) even while
its night screen was black. The off window releases the webcam overnight;
the idle throttle slows the tracker while no hand is in view.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from kinectknob.config import load_config
from kinectknob.schedule import OffWindow
from kinectknob.state import SharedState


def local(y, mo, d, h, mi):
    """Aware datetime at this wall-clock time in the host's zone."""
    return datetime(y, mo, d, h, mi).astimezone()


def _zone(name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - Windows Python ships no tzdata
        pytest.skip("no IANA time zone database on this host")


# -- window arithmetic (host local time, the wall-pad default) ------------------
@pytest.mark.parametrize("h,mi,off", [
    (0, 59, False), (1, 0, True), (3, 30, True), (5, 59, True),
    (6, 0, False), (12, 0, False), (23, 59, False),
])
def test_same_day_window_in_local_time(h, mi, off):
    w = OffWindow("01:00-06:00")
    assert w.tz_name == "local"
    assert w.active(local(2026, 10, 30, h, mi)) is off


@pytest.mark.parametrize("h,mi,off", [
    (19, 59, False), (20, 0, True), (23, 30, True), (0, 0, True),
    (6, 29, True), (6, 30, False), (12, 0, False),
])
def test_overnight_window_crosses_midnight(h, mi, off):
    w = OffWindow("20:00-06:30")
    assert w.active(local(2026, 10, 26, h, mi)) is off


def test_next_change_is_the_window_edge_ahead():
    w = OffWindow("01:00-06:00")
    assert w.next_change(local(2026, 10, 30, 3, 0)) == local(2026, 10, 30, 6, 0)
    assert w.next_change(local(2026, 10, 30, 9, 0)) == local(2026, 10, 31, 1, 0)
    assert w.next_change(local(2026, 10, 30, 0, 30)) == local(2026, 10, 30, 1, 0)


def test_named_zone_survives_dst():
    """06:30 Eastern is 10:30 UTC in summer and 11:30 UTC in winter."""
    _zone("America/New_York")
    utc = _zone("UTC")
    w = OffWindow("20:00-06:30", "America/New_York")
    assert not w.active(datetime(2026, 7, 1, 10, 31, tzinfo=utc))
    assert w.active(datetime(2026, 12, 1, 11, 29, tzinfo=utc))
    assert not w.active(datetime(2026, 12, 1, 11, 31, tzinfo=utc))


def test_empty_spec_disables():
    w = OffWindow("")
    assert not w.enabled and not w.active() and w.next_change() is None
    assert w.describe()["off_now"] is False


def test_naive_now_is_rejected():
    with pytest.raises(ValueError):
        OffWindow("01:00-06:00").active(datetime(2026, 10, 30, 3, 0))


# -- config wiring -------------------------------------------------------------
@pytest.mark.parametrize("bad", ["20:00", "8pm-6am", "20:00-20:00", "25:00-06:00"])
def test_bad_spec_fails_at_config_load(bad, monkeypatch):
    monkeypatch.setenv("KK_OFF_WINDOW", bad)
    with pytest.raises(ValueError):
        load_config()


def test_env_wires_schedule_and_idle(monkeypatch):
    monkeypatch.setenv("KK_OFF_WINDOW", "01:00-06:00")
    monkeypatch.setenv("KK_IDLE_FPS", "6")
    monkeypatch.setenv("KK_IDLE_AFTER_S", "2.5")
    cfg = load_config()
    assert cfg.schedule.off_window == "01:00-06:00"
    assert cfg.schedule.tz == ""
    assert cfg.capture.idle_fps == 6.0
    assert cfg.capture.idle_after_s == 2.5


def test_defaults_leave_both_off(monkeypatch):
    for var in ("KK_OFF_WINDOW", "KK_TZ", "KK_IDLE_FPS", "KK_IDLE_AFTER_S"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.schedule.off_window == "" and cfg.capture.idle_fps == 0.0


def test_negative_idle_fps_clamps_to_off(monkeypatch):
    monkeypatch.setenv("KK_IDLE_FPS", "-3")
    assert load_config().capture.idle_fps == 0.0


# -- health while off ----------------------------------------------------------
def test_scheduled_off_is_healthy():
    """The wall-pad watchdog restarts the node whenever /healthz is not ok;
    an intentionally closed camera must not trip it."""
    s = SharedState()
    assert not s.healthy()          # no frames yet
    s.camera_off = "scheduled off until 06:00"
    assert s.healthy()
    assert s.state_dict()["camera_off"]
