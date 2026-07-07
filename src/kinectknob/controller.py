"""Maps gesture events to Home Assistant media_player service calls.

Volume strategy (the part that makes the knob feel like hardware):

* On **engage** we anchor at the entity's current volume (from the HA client's
  live state cache — no round trip needed).
* Every **turn** computes an absolute target:
      target = anchor + rotation_deg / full_scale_deg
  and stores it as "pending". A coalescer task ships the latest pending value
  at most every ``ha.send_interval_s`` seconds via ``media_player.volume_set``.
  Absolute targets mean dropped/rate-limited intermediate sends cost nothing —
  the next send lands exactly where your hand is. No drift, no runaway.
* On **release** the final target is flushed immediately.
* If HA doesn't know the volume (entity unavailable), we degrade to relative
  ``volume_up``/``volume_down`` detents so the knob still works.

Transport events (swipe/play-pause) are fired immediately, never queued behind
volume traffic.

With no HA configured the controller runs in **dry-run**: everything works
against a simulated volume so gestures can be tuned on a laptop with no HA.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

from .config import AppConfig
from .ha.client import HAClient
from .types import GestureEvent, KnobEngage, KnobRelease, KnobTurn, PlayPauseHold, Swipe

log = logging.getLogger("kk.ctl")

# The Bose Music family (Soundbar 700) uses integer 0-100 volume internally,
# so quantise to 0.01 — sub-step sends would be no-ops that just add traffic.
VOLUME_QUANTUM = 0.01


class Controller:
    def __init__(self, cfg: AppConfig, ha: Optional[HAClient]):
        self.cfg = cfg
        self.ha = ha
        self.dry_run = ha is None
        self._queue: asyncio.Queue[GestureEvent] = asyncio.Queue(maxsize=256)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._anchor: Optional[float] = None       # volume at engage
        self._pending: Optional[float] = None      # latest desired volume
        self._last_sent: Optional[float] = None
        self._last_send_t = 0.0
        self._engaged = False
        self._relative_accum = 0.0                 # degrees, for relative fallback
        self._sim_volume = 0.5                     # dry-run simulated volume
        self._next_overflow_log = 0.0
        self._next_safety_log = 0.0
        self.events_log: deque[str] = deque(maxlen=30)

    # ------------------------------------------------------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def submit(self, events: list[GestureEvent]) -> None:
        """Thread-safe: called from the vision thread."""
        if not events or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue, list(events))
        except RuntimeError:
            return  # loop shutting down

    def _enqueue(self, events: list[GestureEvent]) -> None:
        """Runs on the loop. A full queue must drop quietly, not raise into
        asyncio's callback handler (which logs a traceback per event — a log
        storm under sustained overload). Dropping a KnobTurn is safe: targets
        are absolute, the next one lands exactly where the hand is."""
        for ev in events:
            try:
                self._queue.put_nowait(ev)
            except asyncio.QueueFull:
                now = time.monotonic()
                if now >= self._next_overflow_log:
                    log.warning("event queue full — dropping gesture events (HA slow/stalled?)")
                    self._next_overflow_log = now + 10.0

    async def run(self) -> None:
        sender = asyncio.create_task(self._volume_sender())
        try:
            while True:
                ev = await self._queue.get()
                try:
                    await self._handle(ev)
                except Exception:  # noqa: BLE001
                    log.exception("error handling %s", type(ev).__name__)
        finally:
            sender.cancel()

    # ------------------------------------------------------------------
    async def _handle(self, ev: GestureEvent) -> None:
        if isinstance(ev, KnobEngage):
            self._engaged = True
            self._relative_accum = 0.0
            self._anchor = self._anchor_volume()
            self._last_sent = None   # dedup must never span grips
            self._pending = None
            if self._anchor is None:
                log.info("knob engaged (no known volume -> relative mode)")
                self._log_event("knob engaged (relative mode)")
            else:
                log.info("knob engaged at volume %.2f", self._anchor)
                self._log_event(f"knob engaged @ {self._anchor:.0%}")

        elif isinstance(ev, KnobTurn):
            if not self._engaged:
                return
            if self._anchor is not None:
                target = self._anchor + ev.deg / self.cfg.knob.full_scale_deg
                self._pending = min(max(target, 0.0), self.cfg.ha.max_volume)
            else:
                await self._relative_turn(ev)

        elif isinstance(ev, KnobRelease):
            self._engaged = False
            await self._flush_volume(force=True)
            final = self._pending if self._pending is not None else self._current_volume()
            self._log_event(
                f"knob released ({ev.deg:+.0f} deg -> {final:.0%})" if final is not None
                else f"knob released ({ev.deg:+.0f} deg)"
            )
            self._anchor = None
            self._pending = None

        elif isinstance(ev, Swipe):
            service = "media_next_track" if ev.direction > 0 else "media_previous_track"
            self._log_event("next track" if ev.direction > 0 else "previous track")
            await self._call("media_player", service, self.cfg.ha.media_entity)

        elif isinstance(ev, PlayPauseHold):
            self._log_event("play/pause")
            await self._call("media_player", "media_play_pause", self.cfg.ha.media_entity)

    async def _relative_turn(self, ev: KnobTurn) -> None:
        """No anchor available: emit volume_up/down per detent of rotation.

        Up-detents are safety-gated: never step up while the volume is unknown,
        and never past max_volume. Unlike volume_set, volume_up is uncapped on
        the HA side — blind ups are how a knob blasts a room (2026-07-06)."""
        detent_deg = self.cfg.knob.full_scale_deg * self.cfg.ha.volume_step
        self._relative_accum += ev.delta_deg
        while self._relative_accum >= detent_deg:
            self._relative_accum -= detent_deg
            vol = self._current_volume()
            if vol is None:
                now = time.monotonic()
                if now >= self._next_safety_log:
                    log.info("volume unknown — skipping volume_up detent (safety)")
                    self._next_safety_log = now + 10.0
                continue
            # The device chooses volume_up's step size, so block any step that
            # could overshoot: stop one configured step short of the ceiling.
            if vol >= self.cfg.ha.max_volume - self.cfg.ha.volume_step:
                continue
            await self._call("media_player", "volume_up", self.cfg.ha.volume_entity)
        while self._relative_accum <= -detent_deg:
            self._relative_accum += detent_deg
            await self._call("media_player", "volume_down", self.cfg.ha.volume_entity)

    # ------------------------------------------------------------------
    async def _volume_sender(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.ha.send_interval_s / 2)
            await self._flush_volume()

    async def _flush_volume(self, force: bool = False) -> None:
        if self._pending is None:
            return
        now = time.monotonic()
        if not force and now - self._last_send_t < self.cfg.ha.send_interval_s:
            return
        target = round(round(self._pending / VOLUME_QUANTUM) * VOLUME_QUANTUM, 2)
        if self._last_sent is not None and abs(target - self._last_sent) < VOLUME_QUANTUM / 2:
            return  # same quantised step — sending it again is a no-op
        self._last_send_t = now
        self._last_sent = target
        await self._call(
            "media_player", "volume_set", self.cfg.ha.volume_entity, {"volume_level": target}
        )

    async def _call(self, domain: str, service: str, entity: str, data: Optional[dict] = None) -> None:
        if self.dry_run:
            if service == "volume_set" and data:
                self._sim_volume = data["volume_level"]
            elif service == "volume_up":
                self._sim_volume = min(1.0, self._sim_volume + 0.02)
            elif service == "volume_down":
                self._sim_volume = max(0.0, self._sim_volume - 0.02)
            log.info("[dry-run] %s.%s %s %s", domain, service, entity or "(no entity)", data or "")
            return
        if not entity:
            log.warning("no entity configured for %s.%s", domain, service)
            return
        await self.ha.call_service(domain, service, entity, data)

    # ------------------------------------------------------------------
    async def manual(self, action: str) -> bool:
        """Manual actions from the web UI (for wiring/latency tests)."""
        vol_ent, med_ent = self.cfg.ha.volume_entity, self.cfg.ha.media_entity
        table = {
            "volume_up": ("volume_up", vol_ent, None),
            "volume_down": ("volume_down", vol_ent, None),
            "next": ("media_next_track", med_ent, None),
            "prev": ("media_previous_track", med_ent, None),
            "play_pause": ("media_play_pause", med_ent, None),
        }
        if action not in table:
            return False
        service, entity, data = table[action]
        self._log_event(f"manual: {action}")
        await self._call("media_player", service, entity, data)
        return True

    # ------------------------------------------------------------------
    def _current_volume(self) -> Optional[float]:
        if self.dry_run:
            return self._sim_volume
        return self.ha.volume_level(self.cfg.ha.volume_entity)

    def _anchor_volume(self) -> Optional[float]:
        """Best-known volume for a new grip. On a quick regrip the HA state
        cache may not have echoed our own last volume_set yet — prefer what we
        sent, unless the cache changed AFTER our send (someone used the Bose
        app/remote), in which case the cache is the truth."""
        if self.dry_run:
            return self._sim_volume
        cache = self.ha.volume_level(self.cfg.ha.volume_entity)
        if self._last_sent is None:
            return cache
        cache_t = self.ha.state_updated_at(self.cfg.ha.volume_entity)
        if cache is None or cache_t is None or self._last_send_t > cache_t:
            return self._last_sent
        return cache

    def _log_event(self, text: str) -> None:
        self.events_log.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def snapshot(self) -> dict:
        vol = self._pending if self._pending is not None else self._current_volume()
        return {
            "mode": "dry-run" if self.dry_run else "home-assistant",
            "ha_connected": bool(self.ha and self.ha.connected),
            "ha_error": self.ha.last_error if self.ha else "",
            "volume": vol,
            "engaged": self._engaged,
            "volume_entity": self.cfg.ha.volume_entity,
            "media_entity": self.cfg.ha.media_entity,
            "media_state": self.ha.entity_state(self.cfg.ha.media_entity) if self.ha else None,
            "events": list(self.events_log),
        }
