"""Frame-stream plumbing for the Kinect v2 backend.

Why this exists: the ``freenect2`` binding's default frame listener is a
16-slot queue whose C-callback does a bare ``put_nowait`` — when the consumer
falls behind (90 frames/s arrive across color+depth+IR), every subsequent
frame raises ``queue.Full`` *inside the cffi callback*, and cffi prints a
full traceback per frame to stderr. At ~90 tracebacks/s that log storm is
heavy enough to stagger the whole host (observed as server-wide hitching,
worst while hands are up because tracking load is what tips the consumer
over the edge).

Two defenses, both here because they're pure stdlib and unit-testable:

* ``LatestQueueListener`` — a drop-in replacement listener that silently
  drops the OLDEST queued frame instead of raising. Never propagates an
  exception into the C callback.
* ``read_latest`` — consumer-side drain: block until every needed stream has
  delivered, but always sweep the backlog and keep only the newest frame of
  each type. Newest-wins, same policy as the vision loop's frame slot, so a
  slow cycle costs staleness of one frame, never a growing backlog.
"""
from __future__ import annotations

import time
from queue import Empty, Full, Queue
from typing import Callable


class LatestQueueListener:
    """freenect2-compatible frame listener (has ``__call__`` and ``get``).

    ``__call__`` runs on the binding's C callback thread and must NEVER raise:
    on overflow it drops the oldest queued frame and counts it in ``dropped``.
    """

    def __init__(self, maxsize: int = 16):
        self.queue: Queue = Queue(maxsize=maxsize)
        self.dropped = 0

    def __call__(self, frame_type, frame) -> None:
        try:
            self.queue.put_nowait((frame_type, frame))
            return
        except Full:
            pass
        try:
            self.queue.get_nowait()          # make room: oldest frame loses
            self.dropped += 1
        except Empty:                        # racing consumer emptied it
            pass
        try:
            self.queue.put_nowait((frame_type, frame))
        except Full:                         # racing producer refilled it
            self.dropped += 1

    def get(self, timeout=False):
        return self.queue.get(True, timeout)


def read_latest(
    device,
    no_frame_exc: type[Exception],
    needed: frozenset,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Return ``{frame_type: newest_frame}`` covering every type in ``needed``.

    Blocks on the device until each needed stream has delivered at least one
    frame, opportunistically draining whatever else is queued so only the
    newest frame of each type is kept. Raises ``TimeoutError`` if ``needed``
    can't be covered within ``timeout_s`` (the caller maps this to a
    CaptureError so the container restart policy can recover a USB stall).
    """
    latest: dict = {}
    deadline = clock() + timeout_s
    while True:
        try:
            ftype, frame = device.get_next_frame(timeout=1.0)
            latest[ftype] = frame
        except no_frame_exc:
            if clock() > deadline:
                if latest:
                    raise TimeoutError(
                        f"frame pairing timed out (missing: {set(needed) - latest.keys()})"
                    ) from None
                raise TimeoutError("no frames from device (USB stall?)") from None
            continue
        # Sweep the backlog without blocking; newest of each type wins.
        try:
            while True:
                ftype, frame = device.get_next_frame(timeout=0)
                latest[ftype] = frame
        except no_frame_exc:
            pass
        if needed <= latest.keys():
            return latest
        if clock() > deadline:
            raise TimeoutError(
                f"frame pairing timed out (missing: {set(needed) - latest.keys()})"
            )
