"""Scheduled off window + real camera teardown (2026-10-26).

Two failures, one evening: the overnight USB-stall storms (50-185 recoveries an
hour between 21:00 and 07:00 Eastern) and an in-process recycle that never
actually closed the device, so libfreenect2 handed back the same half-dead
handle ("device @4:5 is already be open!") and every stall cascaded into a
container restart. The sensor's lights flash on every one of those reopens.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from kinectknob.config import load_config
from kinectknob.schedule import OffWindow

NY = ZoneInfo("America/New_York")


def at(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=NY)


# -- window arithmetic ---------------------------------------------------------
@pytest.mark.parametrize("h,mi,off", [
    (19, 59, False), (20, 0, True), (23, 30, True), (0, 0, True),
    (6, 29, True), (6, 30, False), (12, 0, False),
])
def test_overnight_window_crosses_midnight(h, mi, off):
    w = OffWindow("20:00-06:30", "America/New_York")
    assert w.active(at(2026, 10, 26, h, mi)) is off


def test_same_day_window():
    w = OffWindow("13:00-14:00", "America/New_York")
    assert w.active(at(2026, 10, 26, 13, 30))
    assert not w.active(at(2026, 10, 26, 14, 0))
    assert not w.active(at(2026, 10, 26, 3, 0))


def test_next_change_is_the_window_edge_ahead():
    w = OffWindow("20:00-06:30", "America/New_York")
    # in the evening part of the window -> tomorrow 06:30
    assert w.next_change(at(2026, 10, 26, 22, 0)) == at(2026, 10, 27, 6, 30)
    # after midnight -> this morning 06:30
    assert w.next_change(at(2026, 10, 27, 2, 0)) == at(2026, 10, 27, 6, 30)
    # daytime -> tonight 20:00
    assert w.next_change(at(2026, 10, 26, 9, 0)) == at(2026, 10, 26, 20, 0)


def test_wall_clock_survives_dst():
    """06:30 Eastern is 10:30 UTC in summer and 11:30 UTC in winter; the
    window is wall-clock, so both are the edge."""
    w = OffWindow("20:00-06:30", "America/New_York")
    utc = ZoneInfo("UTC")
    assert not w.active(datetime(2026, 7, 1, 10, 31, tzinfo=utc))
    assert w.active(datetime(2026, 12, 1, 11, 29, tzinfo=utc))
    assert not w.active(datetime(2026, 12, 1, 11, 31, tzinfo=utc))


def test_empty_spec_disables():
    w = OffWindow("", "America/New_York")
    assert not w.enabled and not w.active() and w.next_change() is None


@pytest.mark.parametrize("bad", ["20:00", "8pm-6am", "20:00-20:00", "25:00-06:00"])
def test_bad_spec_fails_at_config_load(bad, monkeypatch):
    monkeypatch.setenv("KK_OFF_WINDOW", bad)
    with pytest.raises(ValueError):
        load_config()


def test_env_wires_schedule(monkeypatch):
    monkeypatch.setenv("KK_OFF_WINDOW", "20:00-06:30")
    monkeypatch.setenv("KK_TZ", "America/New_York")
    cfg = load_config()
    assert cfg.schedule.off_window == "20:00-06:30"
    assert cfg.schedule.tz == "America/New_York"


# -- teardown really closes the device ----------------------------------------
def test_stop_only_stops_the_stream():
    pytest.importorskip("freenect2")
    from kinectknob.capture.kinect_v2 import KinectV2Capture

    calls = []

    class Dev:
        def stop(self):
            calls.append("stop")

        def close(self):
            calls.append("close")

    cap = KinectV2Capture.__new__(KinectV2Capture)
    cap._device, cap._listener = Dev(), object()
    cap._streaming, cap._wedged = True, False
    cap.stop()
    assert calls == ["stop"]            # never close(): it segfaults (exit 139)
    assert cap._device is not None      # listeners must outlive native threads
    cap.stop()                          # idempotent
    assert calls == ["stop"]


def test_half_started_device_is_wedged_and_never_closed(monkeypatch):
    """libfreenect2's start() returning false leaves colour transfers in
    flight: closing would free them under libusb, reopening hits BUSY."""
    freenect2 = pytest.importorskip("freenect2")
    from kinectknob.capture import DeviceWedged
    from kinectknob.capture.kinect_v2 import KinectV2Capture

    class Lib:
        @staticmethod
        def freenect2_device_start(_):
            return 0

    class Dev:
        _c_object = object()

        def close(self):
            raise AssertionError("must not close a wedged device")

    monkeypatch.setattr(freenect2, "lib", Lib)
    cap = KinectV2Capture.__new__(KinectV2Capture)
    cap._device, cap._streaming, cap._wedged = Dev(), False, False
    with pytest.raises(DeviceWedged):
        cap._start_streams()
    cap.stop()   # no-op, no close
    assert cap._wedged and cap._device is not None


def test_failure_exits_are_remembered_and_expire(tmp_path):
    import time

    from kinectknob.main import App

    app = App.__new__(App)
    app._exit_log = tmp_path / "capture_exits.json"
    assert app._recent_failure_exits() == []
    app._record_failure_exit()
    app._record_failure_exit()
    assert len(app._recent_failure_exits()) == 2 and app.exit_code == 3
    app._exit_log.write_text('{"exits": [%f]}' % (time.time() - 7200))
    assert app._recent_failure_exits() == []
    app._exit_log.write_text("garbage")
    assert app._recent_failure_exits() == []


# -- endpoints say "camera off", never serve stale frames ---------------------
def _route(app, path, method="GET"):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in r.methods:
            return r.endpoint
    raise KeyError(path)


def _body(resp):
    return resp if isinstance(resp, dict) else json.loads(resp.body)


@pytest.fixture
def web():
    pytest.importorskip("fastapi")
    import numpy as np

    from kinectknob.config import AppConfig
    from kinectknob.controller import Controller
    from kinectknob.state import SharedState
    from kinectknob.web.server import create_app

    cfg = AppConfig()
    shared = SharedState()
    shared.update_fullres(np.zeros((4, 4, 3), np.uint8))  # a leftover frame
    app = create_app(cfg, shared, Controller(cfg, None))
    return app, shared


def test_snapshot_refuses_while_camera_off(web):
    app, shared = web
    shared.set_camera("off", "scheduled off until 06:30")
    resp = asyncio.run(_route(app, "/api/snapshot")())
    assert resp.status_code == 503
    assert _body(resp)["camera"] == "off"


def test_cached_snapshot_never_serves_a_stale_frame(web):
    app, shared = web
    shared.set_camera("ok")
    shared._fullres_t -= 60  # camera stopped feeding a minute ago
    resp = asyncio.run(_route(app, "/api/snapshot")())
    assert resp.status_code == 404


def test_region_queries_report_camera_state(web):
    app, shared = web
    shared.set_camera("off", "scheduled off")
    resp = asyncio.run(_route(app, "/api/presence")(x1=0, y1=0, x2=10, y2=10))
    assert resp.status_code == 503 and _body(resp)["camera"] == "off"
    resp = asyncio.run(_route(app, "/api/region_depth")(x1=0, y1=0, x2=10, y2=10))
    assert resp.status_code == 503


def test_healthz_is_healthy_while_scheduled_off(web):
    app, shared = web
    shared.set_camera("off", "scheduled off")
    assert _body(asyncio.run(_route(app, "/healthz")()))["status"] == "scheduled off"


def test_restart_endpoint(web):
    app, shared = web
    shared.restart_fn = lambda: True
    assert _body(asyncio.run(_route(app, "/api/camera/restart", "POST")()))["ok"]
    shared.restart_fn = lambda: False
    assert asyncio.run(_route(app, "/api/camera/restart", "POST")()).status_code == 409


# -- logical unplug / replug ----------------------------------------------------
def _fake_sysfs(tmp_path):
    for name, pid in (("4-1", "02d9"), ("4-1.1", "02c4"), ("3-2", "0002")):
        d = tmp_path / name
        d.mkdir()
        (d / "idVendor").write_text("045e\n" if pid != "0002" else "051d\n")
        (d / "idProduct").write_text(pid + "\n")
        (d / "authorized").write_text("1\n")
    (tmp_path / "4-1:1.0").mkdir()   # interface dirs have no authorized file
    return tmp_path


def test_replug_targets_only_the_kinect_hub(tmp_path):
    from kinectknob.capture import usb_power

    root = _fake_sysfs(tmp_path)
    assert [p.name for p in usb_power.kinect_nodes(root)] == ["4-1"]
    assert usb_power.set_authorized(False, root)
    assert (root / "4-1" / "authorized").read_text() == "0"
    assert (root / "3-2" / "authorized").read_text() == "1\n"   # UPS untouched
    assert usb_power.replug(off_s=0, root=root)
    assert (root / "4-1" / "authorized").read_text() == "1"


def test_replug_is_a_noop_without_the_mount(tmp_path):
    from kinectknob.capture import usb_power

    assert usb_power.kinect_nodes(tmp_path / "missing") == []
    assert usb_power.replug(off_s=0, root=tmp_path / "missing") is False


def test_rest_before_open_backs_off_only_on_repeats():
    from kinectknob.main import _rest_before_open_s

    assert _rest_before_open_s(0) == 0 and _rest_before_open_s(1) == 0
    assert [_rest_before_open_s(n) for n in (2, 3, 4)] == [30, 60, 120]
    assert _rest_before_open_s(20) == 600


def test_failures_replug_once_they_pile_up(tmp_path, monkeypatch):
    from kinectknob import main

    replugs = []
    monkeypatch.setattr(main.App, "_request_replug", lambda self: replugs.append(1))
    app = main.App.__new__(main.App)
    app._exit_log = tmp_path / "capture_exits.json"
    app.exit_code = 0
    app._fail_exit("stall")
    app._fail_exit("stall")
    assert replugs == [] and app.exit_code == main.EXIT_CAPTURE_FAILURE
    app._fail_exit("stall")                      # third in 30 min
    assert replugs == [1]
    app._exit_log.unlink()
    app._fail_exit("wedged", force_replug=True)  # wedged always replugs
    assert replugs == [1, 1]


def _hub_pair(tmp_path, sensor_on="4"):
    """usb4 (SuperSpeed) and usb3 (its USB 2.0 companion), each with a half
    of the Kinect hub; the sensor hangs under whichever half it attached to."""
    for bus, speed in (("4", "5000"), ("3", "480")):
        root = tmp_path / f"usb{bus}"
        hub = root / f"{bus}-1"
        hub.mkdir(parents=True)
        for f, v in (("idVendor", "045e"), ("idProduct", "02d9"), ("authorized", "1"), ("speed", speed)):
            (hub / f).write_text(v + "\n")
        if bus == sensor_on:
            sensor = hub / f"{bus}-1.1"
            sensor.mkdir()
            for f, v in (("idVendor", "045e"), ("idProduct", "02c4"), ("authorized", "1"), ("speed", speed)):
                (sensor / f).write_text(v + "\n")
    return tmp_path / "usb4", tmp_path / "usb3"


def test_replug_covers_both_hub_halves_superspeed_first(tmp_path, monkeypatch):
    from kinectknob.capture import usb_power

    ss, hs = _hub_pair(tmp_path)
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT", str(ss))
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT2", str(hs))
    monkeypatch.setattr(usb_power.time, "sleep", lambda s: None)
    order = []
    real_write = type(ss).write_text

    def spy(self, data, *a, **k):
        if self.name == "authorized":
            order.append((self.parent.name, data))
        return real_write(self, data, *a, **k)
    monkeypatch.setattr(type(ss), "write_text", spy)
    assert [p.name for p in usb_power.kinect_nodes()] == ["4-1", "3-1"]
    assert usb_power.replug(off_s=0)
    assert order == [("4-1", "0"), ("3-1", "0"), ("4-1", "1"), ("3-1", "1")]


def test_a_sensor_on_the_usb2_twin_is_recognised(tmp_path, monkeypatch):
    from kinectknob.capture import usb_power

    ss, hs = _hub_pair(tmp_path, sensor_on="3")
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT", str(ss))
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT2", str(hs))
    assert usb_power.sensor_speeds() == [480] and usb_power.on_usb2()


def test_a_superspeed_sensor_is_fine(tmp_path, monkeypatch):
    from kinectknob.capture import usb_power

    ss, hs = _hub_pair(tmp_path, sensor_on="4")
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT", str(ss))
    monkeypatch.setenv("KK_USB_SYSFS_MOUNT2", str(hs))
    assert usb_power.sensor_speeds() == [5000] and not usb_power.on_usb2()
