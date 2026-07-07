"""Controller volume math in dry-run mode (no Home Assistant needed)."""
import asyncio

from kinectknob.config import AppConfig
from kinectknob.controller import Controller
from kinectknob.types import KnobEngage, KnobRelease, KnobTurn, PlayPauseHold, Swipe


def run(coro):
    return asyncio.run(coro)


def make_controller(**ha_overrides) -> Controller:
    cfg = AppConfig()
    for k, v in ha_overrides.items():
        setattr(cfg.ha, k, v)
    return Controller(cfg, ha=None)  # dry-run


def test_anchor_plus_rotation():
    async def scenario():
        ctl = make_controller()
        ctl._sim_volume = 0.50
        await ctl._handle(KnobEngage(t=0.0))
        # +54 deg on a 270 deg scale = +20% -> 0.70
        await ctl._handle(KnobTurn(t=0.1, deg=54.0, delta_deg=54.0))
        await ctl._flush_volume(force=True)
        await ctl._handle(KnobRelease(t=0.2, deg=54.0))
        return ctl

    ctl = run(scenario())
    assert abs(ctl._sim_volume - 0.70) < 0.011


def test_volume_clamped_to_zero_and_max():
    async def scenario():
        ctl = make_controller(max_volume=0.8)
        ctl._sim_volume = 0.70
        await ctl._handle(KnobEngage(t=0.0))
        await ctl._handle(KnobTurn(t=0.1, deg=270.0, delta_deg=270.0))  # way past max
        await ctl._flush_volume(force=True)
        assert ctl._sim_volume == 0.8      # capped by max_volume

        await ctl._handle(KnobTurn(t=0.2, deg=-500.0, delta_deg=-770.0))
        await ctl._flush_volume(force=True)
        assert ctl._sim_volume == 0.0
        return ctl

    run(scenario())


def test_quantised_to_bose_steps():
    async def scenario():
        ctl = make_controller()
        ctl._sim_volume = 0.500
        await ctl._handle(KnobEngage(t=0.0))
        await ctl._handle(KnobTurn(t=0.1, deg=1.0, delta_deg=1.0))  # 0.37% -> rounds to 0.5
        await ctl._flush_volume(force=True)
        return ctl

    ctl = run(scenario())
    assert ctl._sim_volume == 0.50  # sub-step turn is a quantised no-op


def test_swipe_and_playpause_do_not_crash_without_entities():
    async def scenario():
        ctl = make_controller()
        await ctl._handle(Swipe(t=0.0, direction=1, speed=1.5))
        await ctl._handle(Swipe(t=1.0, direction=-1, speed=1.5))
        await ctl._handle(PlayPauseHold(t=2.0))
        return ctl

    ctl = run(scenario())
    events = " ".join(ctl.events_log)
    assert "next" in events and "previous" in events and "play/pause" in events


def test_turn_without_engage_is_ignored():
    async def scenario():
        ctl = make_controller()
        ctl._sim_volume = 0.5
        await ctl._handle(KnobTurn(t=0.0, deg=100.0, delta_deg=100.0))
        await ctl._flush_volume(force=True)
        return ctl

    ctl = run(scenario())
    assert ctl._sim_volume == 0.5


def test_snapshot_shape():
    ctl = make_controller()
    snap = ctl.snapshot()
    assert snap["mode"] == "dry-run"
    assert "volume" in snap and "events" in snap


def test_submit_delivers_events_through_the_loop():
    async def scenario():
        ctl = make_controller()
        ctl.attach_loop(asyncio.get_running_loop())
        ctl._sim_volume = 0.50
        ctl.submit([KnobEngage(t=0.0), KnobTurn(t=0.1, deg=54.0, delta_deg=54.0)])
        await asyncio.sleep(0)              # let call_soon_threadsafe land
        for _ in range(2):
            await ctl._handle(ctl._queue.get_nowait())
        await ctl._flush_volume(force=True)
        return ctl

    ctl = run(scenario())
    assert abs(ctl._sim_volume - 0.70) < 0.011


class FakeHA:
    """Minimal HAClient stand-in: a device whose volume steps 0.02 per
    volume_up, or one that never reports a volume (volume=None)."""

    def __init__(self, volume=None):
        self._vol = volume
        self.connected = True
        self.last_error = ""
        self.calls = []

    async def call_service(self, domain, service, entity_id, data=None):
        self.calls.append((service, data))
        if service == "volume_up" and self._vol is not None:
            self._vol = min(1.0, self._vol + 0.02)
        elif service == "volume_down" and self._vol is not None:
            self._vol = max(0.0, self._vol - 0.02)
        return True

    def volume_level(self, entity_id):
        return self._vol

    def state_updated_at(self, entity_id):
        return None

    def entity_state(self, entity_id):
        return "playing"


def make_ha_controller(ha, **ha_overrides) -> Controller:
    cfg = AppConfig()
    cfg.ha.volume_entity = "media_player.test_bose"
    cfg.ha.media_entity = "media_player.test_spotify"
    for k, v in ha_overrides.items():
        setattr(cfg.ha, k, v)
    return Controller(cfg, ha=ha)


def test_relative_mode_never_steps_up_blind():
    """Regression (2026-07-06): with the device volume unknown, up-detents used
    to fire uncapped volume_up calls — how the soundbar got blasted."""
    async def scenario():
        ha = FakeHA(volume=None)
        ctl = make_ha_controller(ha, max_volume=0.2)
        await ctl._handle(KnobEngage(t=0.0))          # anchor is None -> relative
        for i in range(1, 40):                        # a long upward twist
            await ctl._handle(KnobTurn(t=i / 30, deg=i * 10.0, delta_deg=10.0))
        return ha

    ha = run(scenario())
    assert not any(s == "volume_up" for s, _ in ha.calls)


def test_relative_mode_ups_stop_at_max_volume():
    async def scenario():
        ha = FakeHA(volume=0.15)
        ctl = make_ha_controller(ha, max_volume=0.2)
        await ctl._handle(KnobEngage(t=0.0))
        ctl._anchor = None                            # force relative mode
        for i in range(1, 40):
            await ctl._handle(KnobTurn(t=i / 30, deg=i * 10.0, delta_deg=10.0))
        return ha

    ha = run(scenario())
    ups = [s for s, _ in ha.calls if s == "volume_up"]
    # 0.15 -> 0.17 -> 0.19; the next step could overshoot the 0.2 cap: blocked.
    assert len(ups) == 2
    assert ha._vol <= 0.2


def test_relative_mode_downs_always_allowed():
    async def scenario():
        ha = FakeHA(volume=None)                      # even blind, down is safe
        ctl = make_ha_controller(ha, max_volume=0.2)
        await ctl._handle(KnobEngage(t=0.0))
        for i in range(1, 40):
            await ctl._handle(KnobTurn(t=i / 30, deg=-i * 10.0, delta_deg=-10.0))
        return ha

    ha = run(scenario())
    assert any(s == "volume_down" for s, _ in ha.calls)


def test_submit_overflow_drops_quietly():
    async def scenario():
        ctl = make_controller()
        ctl.attach_loop(asyncio.get_running_loop())
        # Fill the queue to the brim, then overflow it hard: nothing may raise
        # (a raise here would reach asyncio's callback exception handler and
        # log a traceback per event — the storm this guards against).
        events = [KnobTurn(t=i / 30, deg=float(i), delta_deg=1.0) for i in range(400)]
        ctl.submit(events)
        await asyncio.sleep(0)
        assert ctl._queue.qsize() == 256    # capacity, not 400
        return ctl

    run(scenario())
