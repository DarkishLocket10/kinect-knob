"""Regression tests for review findings — each test encodes a fixed bug."""
import asyncio

import numpy as np

from conftest import Timeline, make_hand
from kinectknob.config import AppConfig, load_config
from kinectknob.controller import Controller
from kinectknob.gestures.engine import GestureEngine
from kinectknob.types import KnobEngage, KnobRelease, KnobTurn


# ---------------------------------------------------------------------------
# Engine: a different hand must never inherit an engaged knob.
# ---------------------------------------------------------------------------
def test_second_hand_cannot_steal_engaged_knob(cfg):
    tl = Timeline(GestureEngine(cfg))
    # Hand A engages on the left side of the frame.
    tl.step([make_hand(pose="pinch", center=(160, 300))], n=6)
    assert tl.engine.state == GestureEngine.ENGAGED
    tl.events.clear()
    # Hand A drops out; hand B (far away, bigger) appears during the grace
    # window with a neutral-ish pose.
    handB = make_hand(pose="pinch", center=(520, 300), scale=1.3)
    evs = tl.step([handB], n=3)
    turns = [e for e in evs if isinstance(e, KnobTurn)]
    assert not turns, "another hand must not produce knob turns"
    # With hand A gone for good, the grip must release rather than transfer.
    tl.step([handB], n=15)
    release_idx = next(
        i for i, e in enumerate(tl.events) if isinstance(e, KnobRelease)
    )
    # No turns before the release — B never drove A's grip.
    assert not any(isinstance(e, KnobTurn) for e in tl.events[:release_idx])
    # B may then start its own FRESH grip (anyone in range may use the knob),
    # but it must begin from zero — never inheriting A's accumulated rotation.
    if any(isinstance(e, KnobEngage) for e in tl.events[release_idx:]):
        assert abs(tl.engine.snapshot().angle_deg) < 2.0
    else:
        assert tl.engine.state == GestureEngine.IDLE


# ---------------------------------------------------------------------------
# Engine: rotation during a tracking dropout is re-based, not applied as a jump.
# ---------------------------------------------------------------------------
def test_dropout_rotation_is_rebased_not_jumped(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch", angle_deg=0.0)], n=6)
    assert tl.engine.state == GestureEngine.ENGAGED
    # 5 blind frames (~0.17 s, inside the 0.3 s grace) during which the hand
    # rotated 30 degrees.
    tl.step([], n=5)
    assert tl.engine.state == GestureEngine.ENGAGED
    tl.step([make_hand(pose="pinch", angle_deg=30.0)], n=15)
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    final = turns[-1].deg if turns else 0.0
    # The 30 deg happened while blind: it must NOT appear in the accumulator.
    assert abs(final) < 5.0


# ---------------------------------------------------------------------------
# Engine: unmirrored view compensates rotation and swipe direction.
# ---------------------------------------------------------------------------
def test_unmirrored_rotation_sign_compensated(cfg):
    cfg.capture.mirror = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    # Screen-clockwise rotation in an unmirrored feed = user counter-clockwise.
    for i in range(1, 31):
        tl.step([make_hand(pose="pinch", angle_deg=i * 2.0)])
    tl.step([make_hand(pose="pinch", angle_deg=60.0)], n=15)
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert turns[-1].deg < -50.0, "unmirrored view must flip rotation sign"


def test_unmirrored_swipe_direction_compensated(cfg):
    from kinectknob.types import Swipe

    cfg.capture.mirror = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="two", center=(180, 300))], n=15)
    for i in range(1, 7):  # moves right on screen = user's LEFT in raw view
        tl.step([make_hand(pose="two", center=(180 + i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == -1


# ---------------------------------------------------------------------------
# Controller: quick regrip anchors on our own last-sent volume, not stale cache.
# ---------------------------------------------------------------------------
class FakeHA:
    """Stale-cache HA stand-in: volume_level never reflects our sends."""

    def __init__(self, volume=0.50, updated_at=0.0):
        self._volume = volume
        self._updated_at = updated_at
        self.connected = True
        self.last_error = ""
        self.calls = []

    async def call_service(self, domain, service, entity_id, data=None):
        self.calls.append((service, data))
        return True

    def volume_level(self, entity_id):
        return self._volume

    def state_updated_at(self, entity_id):
        return self._updated_at

    def entity_state(self, entity_id):
        return "playing"


def _cfg_with_entities() -> AppConfig:
    cfg = AppConfig()
    cfg.ha.volume_entity = "media_player.bose"
    cfg.ha.media_entity = "media_player.spotify"
    return cfg


def test_quick_regrip_uses_last_sent_anchor():
    async def scenario():
        ha = FakeHA(volume=0.50, updated_at=0.0)  # cache primed long ago, never updates
        ctl = Controller(_cfg_with_entities(), ha)
        # Grip 1: 0.50 + 54/270 = 0.70
        await ctl._handle(KnobEngage(t=0.0))
        await ctl._handle(KnobTurn(t=0.1, deg=54.0, delta_deg=54.0))
        await ctl._handle(KnobRelease(t=0.2, deg=54.0))
        # Quick regrip before HA echoes the state change: anchor must be 0.70.
        await ctl._handle(KnobEngage(t=0.5))
        await ctl._handle(KnobTurn(t=0.6, deg=54.0, delta_deg=54.0))
        await ctl._handle(KnobRelease(t=0.7, deg=54.0))
        return ha

    ha = run_async(scenario())
    volume_sets = [d["volume_level"] for (s, d) in ha.calls if s == "volume_set"]
    assert volume_sets[-1] == 0.90, f"ratchet must accumulate: {volume_sets}"


def test_external_change_after_send_wins_anchor():
    async def scenario():
        ha = FakeHA(volume=0.50, updated_at=0.0)
        ctl = Controller(_cfg_with_entities(), ha)
        await ctl._handle(KnobEngage(t=0.0))
        await ctl._handle(KnobTurn(t=0.1, deg=54.0, delta_deg=54.0))
        await ctl._handle(KnobRelease(t=0.2, deg=54.0))   # sent 0.70
        # Someone drops it to 0.30 with the Bose app AFTER our send.
        import time as _time

        ha._volume = 0.30
        ha._updated_at = _time.monotonic() + 1.0
        await ctl._handle(KnobEngage(t=0.5))
        await ctl._handle(KnobTurn(t=0.6, deg=27.0, delta_deg=27.0))  # +10%
        await ctl._flush_volume(force=True)
        return ha

    ha = run_async(scenario())
    volume_sets = [d["volume_level"] for (s, d) in ha.calls if s == "volume_set"]
    assert volume_sets[-1] == 0.40, f"external change must win the anchor: {volume_sets}"


def test_dedup_does_not_span_grips():
    async def scenario():
        ha = FakeHA(volume=0.65, updated_at=0.0)
        ctl = Controller(_cfg_with_entities(), ha)
        ctl._last_sent = 0.70          # left over from an earlier grip
        ctl._last_send_t = 0.0         # ...sent before the cache was updated? no:
        ha._updated_at = 1e9           # cache is fresher than our old send
        await ctl._handle(KnobEngage(t=0.0))   # anchor = cache 0.65, dedup reset
        await ctl._handle(KnobTurn(t=0.1, deg=13.5, delta_deg=13.5))  # +5% -> 0.70
        await ctl._flush_volume(force=True)
        return ha

    ha = run_async(scenario())
    volume_sets = [d["volume_level"] for (s, d) in ha.calls if s == "volume_set"]
    assert volume_sets == [0.70], "0.70 must be sent even though it was 'last sent' pre-grip"


def test_volume_step_clamped(monkeypatch):
    monkeypatch.setenv("KK_VOLUME_STEP", "0")
    cfg = load_config(None)
    assert cfg.ha.volume_step >= 0.001


def run_async(coro):
    return asyncio.run(coro)
