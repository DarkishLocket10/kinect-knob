"""Controller volume math in dry-run mode (no Home Assistant needed)."""
import asyncio

from kinectknob.config import AppConfig
from kinectknob.controller import Controller
from kinectknob.types import FistHold, KnobEngage, KnobRelease, KnobTurn, Swipe


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


def test_swipe_and_fist_do_not_crash_without_entities():
    async def scenario():
        ctl = make_controller()
        await ctl._handle(Swipe(t=0.0, direction=1, speed=1.5))
        await ctl._handle(Swipe(t=1.0, direction=-1, speed=1.5))
        await ctl._handle(FistHold(t=2.0))
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
