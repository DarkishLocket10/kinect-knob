"""Knob gesture: engage, rotate, release — against the real engine."""
import numpy as np

from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine, openness, pinch_ratio
from kinectknob.types import KnobEngage, KnobRelease, KnobTurn


def test_pose_classifiers():
    assert openness(make_hand(pose="open")) == "open"
    assert openness(make_hand(pose="fist")) == "fist"
    assert pinch_ratio(make_hand(pose="pinch")) < 0.42
    assert pinch_ratio(make_hand(pose="release")) > 0.65
    assert pinch_ratio(make_hand(pose="open")) > 0.65


def test_engage_requires_debounce(cfg):
    tl = Timeline(GestureEngine(cfg))
    evs = tl.step([make_hand(pose="pinch")], n=1)
    assert not any(isinstance(e, KnobEngage) for e in evs)
    evs = tl.step([make_hand(pose="pinch")], n=4)
    assert any(isinstance(e, KnobEngage) for e in evs)


def test_open_hand_never_engages(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=60)
    assert not any(isinstance(e, KnobEngage) for e in tl.events)


def test_fist_never_engages(cfg):
    # A curled hand's thumb rests against the curled index, so by pinch ratio
    # alone it looks like a grip — the pose gate must reject it (the classic
    # accidental engage: a hand relaxed on the armrest turning the volume).
    fist = make_hand(pose="fist")
    assert pinch_ratio(fist) < cfg.knob.engage_pinch  # would engage without the gate
    tl = Timeline(GestureEngine(cfg))
    tl.step([fist], n=60)
    assert not any(isinstance(e, KnobEngage) for e in tl.events)


def test_low_confidence_hand_ignored(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch", score=0.30)], n=60)
    assert not any(isinstance(e, KnobEngage) for e in tl.events)
    assert "low confidence" in tl.engine.snapshot().gated_out


def test_clockwise_turn_raises_angle(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)  # engage
    # Rotate 60 degrees clockwise over 30 frames, then hold to settle.
    for i in range(1, 31):
        tl.step([make_hand(pose="pinch", angle_deg=i * 2.0)])
    tl.step([make_hand(pose="pinch", angle_deg=60.0)], n=15)

    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert turns, "rotation should emit KnobTurn events"
    final = turns[-1].deg
    # 60 deg minus the 3 deg deadband, One Euro settles at rest.
    assert 50.0 < final < 60.0


def test_counterclockwise_turn_is_negative(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    for i in range(1, 31):
        tl.step([make_hand(pose="pinch", angle_deg=-i * 2.0)])
    tl.step([make_hand(pose="pinch", angle_deg=-60.0)], n=15)
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert turns[-1].deg < -50.0


def test_invert_flips_sign(cfg):
    cfg.knob.invert = True
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    for i in range(1, 31):
        tl.step([make_hand(pose="pinch", angle_deg=i * 2.0)])
    tl.step([make_hand(pose="pinch", angle_deg=60.0)], n=15)
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert turns[-1].deg < -50.0


def test_deadband_swallows_wobble(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    rng = np.random.default_rng(42)
    for _ in range(60):  # 2 seconds of +/-1 deg tremor
        tl.step([make_hand(pose="pinch", angle_deg=float(rng.uniform(-1.0, 1.0)))])
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert all(abs(t.deg) < 2.0 for t in turns)


def test_release_emits_final_angle(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    for i in range(1, 21):
        tl.step([make_hand(pose="pinch", angle_deg=i * 2.0)])
    tl.step([make_hand(pose="pinch", angle_deg=40.0)], n=10)
    evs = tl.step([make_hand(pose="release", angle_deg=40.0)], n=10)
    releases = [e for e in evs if isinstance(e, KnobRelease)]
    assert len(releases) == 1
    assert 30.0 < releases[0].deg < 40.0
    assert tl.engine.state == GestureEngine.IDLE


def test_hand_loss_grace_then_release(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    tl.step([make_hand(pose="pinch", angle_deg=20.0)], n=10)
    # Brief dropout shorter than the grace window: still engaged.
    tl.step([], n=3)
    assert tl.engine.state == GestureEngine.ENGAGED
    # Long dropout: released.
    evs = tl.step([], n=20)
    assert any(isinstance(e, KnobRelease) for e in evs)
    assert tl.engine.state == GestureEngine.IDLE


def test_regrip_ratchets(cfg):
    """Release, rotate back, re-pinch: rotation must restart from zero."""
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    for i in range(1, 21):
        tl.step([make_hand(pose="pinch", angle_deg=i * 2.0)])
    tl.step([make_hand(pose="release", angle_deg=40.0)], n=8)     # let go
    tl.step([make_hand(pose="open", angle_deg=0.0)], n=8)         # wind back
    tl.events.clear()
    tl.step([make_hand(pose="pinch", angle_deg=0.0)], n=5)        # re-grip
    tl.step([make_hand(pose="pinch", angle_deg=0.0)], n=5)
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    assert all(abs(t.deg) < 2.0 for t in turns), "regrip must not inherit old rotation"


def test_glitch_frame_rejected(cfg):
    """A single-frame 90-degree landmark glitch must not move the knob."""
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=5)
    tl.step([make_hand(pose="pinch")], n=5)
    tl.step([make_hand(pose="pinch", angle_deg=90.0)], n=1)   # glitch
    tl.step([make_hand(pose="pinch", angle_deg=0.0)], n=20)   # back to reality
    turns = [e for e in tl.events if isinstance(e, KnobTurn)]
    final = turns[-1].deg if turns else 0.0
    assert abs(final) < 5.0


def test_small_faraway_hand_ignored(cfg):
    tl = Timeline(GestureEngine(cfg))
    tiny = make_hand(pose="pinch", scale=0.15)  # hand size 15px < 4.5% of 480
    tl.step([tiny], n=30)
    assert not tl.events
    assert not tl.engine.snapshot().hand_present


def test_depth_gate_blocks_far_hand(cfg):
    tl = Timeline(GestureEngine(cfg))
    far = lambda x, y: 4.5   # noqa: E731 — beyond depth_max_m
    tl.step([make_hand(pose="pinch")], n=30, depth_sampler=far)
    assert not tl.events

    near = lambda x, y: 1.5  # noqa: E731
    tl.step([make_hand(pose="pinch")], n=5, depth_sampler=near)
    assert any(isinstance(e, KnobEngage) for e in tl.events)


def test_no_engage_while_hand_moving_fast(cfg):
    tl = Timeline(GestureEngine(cfg))
    # Pinch pose racing across the frame: must not engage.
    for i in range(15):
        tl.step([make_hand(pose="pinch", center=(60 + i * 40, 300))])
    assert not any(isinstance(e, KnobEngage) for e in tl.events)
