"""Scheduled camera-off window (2026-10-26).

The Kinect ran 24/7 and its USB stalls clustered overnight: of ~2,600 stall
recoveries logged 2026-10-14..26, the storms of 50-185 per hour all fell
between 21:00 and 07:00 Eastern. Nobody uses the knob at night, so the sensor
is simply switched off then — stream stopped and device closed, so the IR
emitters and the board itself idle and cool down.

``OffWindow("20:00-06:30", "America/New_York")`` is a daily wall-clock window
in a named time zone (DST-correct via zoneinfo). Windows may cross midnight.
An empty spec disables the schedule.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


def _parse_hhmm(text: str) -> dtime:
    hh, _, mm = text.strip().partition(":")
    t = dtime(int(hh), int(mm or 0))
    return t


class OffWindow:
    def __init__(self, spec: str = "", tz: str = "America/New_York"):
        self.spec = (spec or "").strip()
        self.tz = ZoneInfo(tz or "UTC")
        self.tz_name = tz or "UTC"
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
            return datetime.now(self.tz)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(self.tz)

    def active(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        t = self._local(now).time()
        if self.start < self.end:                    # same-day window
            return self.start <= t < self.end
        return t >= self.start or t < self.end       # crosses midnight

    def next_change(self, now: Optional[datetime] = None) -> Optional[datetime]:
        """When the on/off state next flips (aware datetime in the window's
        zone), or None when the schedule is disabled."""
        if not self.enabled:
            return None
        local = self._local(now)
        edge = self.end if self.active(local) else self.start
        cand = datetime.combine(local.date(), edge, tzinfo=self.tz)
        if cand <= local:
            cand = datetime.combine(local.date() + timedelta(days=1), edge,
                                    tzinfo=self.tz)
        return cand

    def describe(self, now: Optional[datetime] = None) -> dict:
        nxt = self.next_change(now)
        return {
            "off_window": self.spec,
            "tz": self.tz_name,
            "off_now": self.active(now),
            "next_change": nxt.isoformat(timespec="minutes") if nxt else None,
        }
