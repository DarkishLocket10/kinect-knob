"""Capture backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import Frame


class CaptureError(RuntimeError):
    """Device missing / failed to open, with a user-actionable message."""


class DeviceWedged(CaptureError):
    """Opened, but left half-started: libfreenect2's start() submitted the
    colour transfers, then every depth/IR submission failed and it returned
    false WITHOUT cancelling them. Its API cannot reclaim them — stop() is a
    no-op on a device it never marked streaming, and close() would free
    transfers that are still in flight. Only process exit releases them (the
    kernel reaps the URBs), so this is the one capture failure that must end
    the process. Reopening in-process is what produced the 20
    LIBUSB_ERROR_BUSY lines — one per stuck colour transfer — before every
    failed restart through 2026-10-26."""


class DeviceAbsent(CaptureError):
    """No camera is physically attached — distinct from one that is attached
    and misbehaving.

    The difference decides recovery policy (see ``main._acquire_capture``):
    a device that is *absent* will not be summoned by restarting the process,
    so the app waits in-process for it to enumerate. Treating the two cases
    alike is what produced the 61-second container boot loop of 2026-09-17,
    where a Kinect that had dropped off the bus made the app fall back to a
    webcam this host does not have, exit, and be restarted a couple of
    thousand times.
    """


class CaptureBase(ABC):
    name = "base"
    has_depth = False

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Block until the next frame (or ~1s timeout -> None). rgb is RGB order,
        NOT mirrored (mirroring is applied centrally in the capture thread)."""

    @abstractmethod
    def stop(self) -> None: ...

    def set_active(self, active: bool) -> None:
        """Tell the backend whether the vision pipeline is consuming frames
        for real (someone is present) or merely idling.

        Backends may use it to drop expensive per-frame work — kinect_v2
        throttles depth registration — but must KEEP STREAMING either way:
        ``/api/snapshot`` serves whiteboard-sync precisely when nobody is
        here, so an idle camera is not an off camera. Default: no-op.
        """
