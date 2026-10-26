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


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def kinect_nodes(root: Path | None = None) -> list[Path]:
    """sysfs device dirs to (de)authorize: the Kinect's hub, or the sensor
    itself if no hub is visible. Empty when the mount is absent."""
    root = root or _root()
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
    for d in nodes:
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
