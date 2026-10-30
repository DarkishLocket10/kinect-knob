"""Scheduled camera-off window.

Ported from the Kinect deployment (main, 2026-10-26) for the Surface wall-pad
node, where the cost is CPU rather than USB stalls: hand tracking ran 24/7 at
~2.3 cores on the clipboard's i7-1065G7 even while the panel's night screen
was black and nobody was there to use it. Inside the window the webcam is
released, so the vision loop has nothing to process.

``OffWindow("01:00-06:00", "America/New_York")`` is a daily wall-clock window
in a named time zone (DST-correct via zoneinfo). Windows may cross midnight.
An empty spec disables the schedule. An empty ``tz`` means the host's local
time - Windows Python ships no IANA database (``tzdata`` is not installed on
the wall pad), and the host clock is already Eastern.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, tzinfo
from typing import Optional


def _parse_hhmm(text: str) -> dtime:
    hh, _, mm = text.strip().partition(":")
    return dtime(int(hh), int(mm or 0))


class OffWindow:
    def __init__(self, spec: str = "", tz: str = ""):
        self.spec = (spec or "").strip()
        self.tz_name = (tz or "").strip() or "local"
        self.tz: Optional[tzinfo] = None
        if tz and tz.strip():
            from zoneinfo import ZoneInfo
            self.tz = ZoneInfo(tz.strip())
        self.start: Optional[dtime] = None
        self.end: Optional[dtime] = None
        if self.spec:
            a, sep, b = self.spec.partition("-")
            try:
                if not sep:
                    raise ValueError
                self.start, self.end = _parse_hhmm(a), _parse_hhmm(b)
            except ValueError:
                raise ValueError(
                    f"off window must look like HH:MM-HH:MM, got {spec!r}") from None
            if self.start == self.end:
                raise ValueError(f"off window {spec!r} is empty")

    @property
    def enabled(self) -> bool:
        return self.start is not None

    def _local(self, now: Optional[datetime]) -> datetime:
        if now is None:
            return datetime.now(self.tz) if self.tz else datetime.now().astimezone()
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(self.tz) if self.tz else now.astimezone()

    def active(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        t = self._local(now).time()
        if self.start < self.end:                    # same-day window
            return self.start <= t < self.end
        return t >= self.start or t < self.end       # crosses midnight

    def next_change(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """When the on/off state next flips (aware datetime), or None when
        the schedule is disabled."""
        if not self.enabled:
            return None
        local = self._local(now)
        edge = self.end if self.active(local) else self.start
        zone = self.tz or local.tzinfo
        cand = datetime.combine(local.date(), edge, tzinfo=zone)
        if cand <= local:
            cand = datetime.combine(local.date() + timedelta(days=1), edge,
                                    tzinfo=zone)
        return cand

    def describe(self, now: Optional[datetime] = None) -> dict:
        nxt = self.next_change(now)
        return {
            "off_window": self.spec,
            "tz": self.tz_name,
            "off_now": self.active(now),
            "next_change": nxt.isoformat(timespec="minutes") if nxt else None,
        }
