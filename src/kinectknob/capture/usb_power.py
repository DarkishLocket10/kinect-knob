"""Logical unplug/replug of the Kinect through sysfs ``authorized``.

Why: on 2026-10-26 the sensor got into a state where every fresh open failed
("all submissions failed" on the depth/IR isochronous transfers) — across
process restarts AND libfreenect2's own USB reset on open. De-authorizing the
Kinect's built-in hub (045e:02d9) and authorizing it again — what the kernel
does for a physical unplug — recovered it every time. That is the "proper
restart"; it is also how the sensor is switched off for the night window
(unconfigured: no streams, emitters dark).

The container sees the host's USB controller sysfs node bind-mounted at
``KK_USB_SYSFS_MOUNT`` (default /host-usb; see docker-compose.yml). Writing an
``authorized`` file needs no extra capability. Without the mount every call
here is a logged no-op, so dev boxes and other hosts are unaffected.

The hub is two devices. A USB 3 hub enumerates twice: its SuperSpeed half on
the controller's USB3 bus (usb4 here, 4-1) and a USB 2.0 twin on the
companion bus (usb3, 3-1), same 045e:02d9. On 2026-10-27 a replug of 4-1
alone brought the sensor back under the 2.0 twin (3-1.1, 480 Mb/s), where it
cannot stream; every later replug toggled the empty SuperSpeed half and the
camera stayed down 5.4 h until somebody physically reseated the cable. So the
companion bus is mounted too (``KK_USB_SYSFS_MOUNT2``, default /host-usb2),
both halves are unplugged together, and the SuperSpeed half is plugged back
FIRST so the sensor attaches there.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger("kk.usb")

_MS = "045e"
_KINECT_HUB = "02d9"      # "NuiSensor Adaptor" — parent of the sensor
_KINECT_SENSOR = "02c4"   # "Xbox NUI Sensor"


def _root() -> Path:
    return Path(os.environ.get("KK_USB_SYSFS_MOUNT", "/host-usb"))


def _roots() -> list[Path]:
    """Every mounted bus: the USB3 one and, if mounted, its USB 2.0 companion."""
    out = [_root(), Path(os.environ.get("KK_USB_SYSFS_MOUNT2", "/host-usb2"))]
    return [r for r in dict.fromkeys(out) if r.is_dir()]


def _speed(d: Path) -> int:
    try:
        return int(float(_read(d / "speed") or 0))
    except ValueError:
        return 0


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def kinect_nodes(root: Path | None = None) -> list[Path]:
    """sysfs device dirs to (de)authorize: the Kinect's hub, or the sensor
    itself if no hub is visible — on ``root``, or on every mounted bus.
    Empty when no mount is present."""
    if root is None:
        return [n for r in _roots() for n in kinect_nodes(r)]
    if not root.is_dir():
        return []
    hubs, sensors = [], []
    for d in sorted(root.iterdir()):
        if not (d / "authorized").is_file() or _read(d / "idVendor") != _MS:
            continue
        pid = _read(d / "idProduct")
        if pid == _KINECT_HUB:
            hubs.append(d)
        elif pid == _KINECT_SENSOR:
            sensors.append(d)
    return hubs or sensors


def set_authorized(on: bool, root: Path | None = None) -> bool:
    nodes = kinect_nodes(root)
    if not nodes:
        log.info("no Kinect sysfs node mounted — cannot %s it at the USB level",
                 "authorize" if on else "de-authorize")
        return False
    ok = True
    if on:
        # SuperSpeed half first, then the 2.0 twin: the sensor attaches to
        # whichever hub half is up, and it can only stream on SuperSpeed.
        nodes = sorted(nodes, key=_speed, reverse=True)
    for i, d in enumerate(nodes):
        if on and i:
            time.sleep(1.0)
        try:
            if _read(d / "authorized") != ("1" if on else "0"):
                (d / "authorized").write_text("1" if on else "0")
                log.info("USB %s %s", d.name, "re-authorized (plugged back in)"
                         if on else "de-authorized (logically unplugged)")
        except OSError as exc:
            log.warning("could not write %s/authorized: %s", d.name, exc)
            ok = False
    return ok


def replug(off_s: float = 10.0, root: Path | None = None) -> bool:
    """Logical unplug, wait, plug back in. Only call while this process does
    not hold the device open (main runs it at process start). 10 s: the one
    replug that held on 2026-10-26 had the sensor off that long; 4 s did not."""
    if not set_authorized(False, root):
        return False
    time.sleep(off_s)
    return set_authorized(True, root)


def sensor_speeds(root: Path | None = None) -> list[int]:
    """Link speeds (Mb/s) of every enumerated Kinect sensor (045e:02c4)."""
    roots = [root] if root is not None else _roots()
    out = []
    for r in roots:
        if not r.is_dir():
            continue
        for d in [*r.glob("*-*"), *r.glob("*-*/*-*")]:
            if ":" in d.name or not (d / "idProduct").is_file():
                continue
            if _read(d / "idVendor") == _MS and _read(d / "idProduct") == _KINECT_SENSOR:
                out.append(_speed(d))
    return out


def on_usb2(root: Path | None = None) -> bool:
    """Is the sensor attached at USB 2.0 speed — present, but unable to
    stream? Only a replug of both hub halves gets it back to SuperSpeed."""
    speeds = sensor_speeds(root)
    return bool(speeds) and max(speeds) < 5000
