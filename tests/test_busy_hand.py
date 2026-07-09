"""Busy-hand rejection: a hand holding an object (water bottle, mug,
toothbrush) must never control anything — and when the other hand is up and
free, the free hand takes over as the controlling hand.

Object evidence is depth-based (object_gap): a held object's surface sits
well in FRONT of the wrist plane over the palm area. Shape alone cannot gate
this — the knob pinch is shaped exactly like a grip on a small object.
"""
from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine, object_gap
from kinectknob.types import KnobEngage, KnobRelease, PlayPauseHold

LEFT, RIGHT = (200.0, 300.0), (450.0, 300.0)


def bottle_left(x, y):
    """An object surface ~12 cm in front of the wrist, over the LEFT hand's
    palm area only. The palm probes sample around y≈280-300; both hands'
    wrists sit near y≈375, safely in the far region."""
    return 0.58 if (x < 320 and y < 340) else 0.70


def flat(x, y):
    return 0.70


def bottle_right(x, y):
    return 0.55 if (x >= 320 and y < 340) else 0.70


def _engages(tl):
    return [e for e in tl.events if isinstance(e, KnobEngage)]


def test_object_gap_reads_the_held_object():
    held = object_gap(make_hand(center=LEFT, pose="pinch"), bottle_left)
    empty = object_gap(make_hand(center=LEFT, pose="pinch"), flat)
    assert held is not None and held > 0.10
    assert empty is not None and abs(empty) < 0.02


def test_holding_hand_cannot_engage_knob(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(center=LEFT, pose="pinch")], n=60, depth_sampler=bottle_left)
    assert not _engages(tl)
    # Put the bottle down: after the busy linger the same grip engages.
    tl.step([make_hand(center=LEFT, pose="pinch")], n=60, depth_sampler=flat)
    assert len(_engages(tl)) == 1


def test_free_hand_beats_bigger_holding_hand(cfg):
    """Size normally picks the primary hand; a free hand must win anyway."""
    holding = make_hand(center=LEFT, pose="pinch", scale=1.3)
    free = make_hand(center=RIGHT, pose="pinch")
    tl = Timeline(GestureEngine(cfg))
    tl.step([holding, free], n=15, depth_sampler=bottle_left)
    assert _engages(tl)
    assert tl.engine.snapshot().palm_xy[0] > 320   # the free (right) hand


def test_free_hand_steals_primary_from_holding_hand(cfg):
    """Bottle hand is up first and owns tracking; raising the other hand
    hands control over to it (sticky selection must not keep the bottle)."""
    holding = make_hand(center=LEFT, pose="open")
    tl = Timeline(GestureEngine(cfg))
    tl.step([holding], n=30, depth_sampler=bottle_left)
    assert tl.engine.snapshot().palm_xy[0] < 320
    assert not tl.events                           # busy hand does nothing
    tl.step([holding, make_hand(center=RIGHT, pose="pinch")],
            n=15, depth_sampler=bottle_left)
    assert tl.engine.snapshot().palm_xy[0] > 320
    assert _engages(tl)


def test_object_appearing_mid_grip_does_not_release(cfg):
    """Release stays the pinch's job: depth noise reading as a held object
    while twisting must not drop the grip."""
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(center=RIGHT, pose="pinch")], n=20, depth_sampler=flat)
    assert len(_engages(tl)) == 1
    tl.step([make_hand(center=RIGHT, pose="pinch")], n=30, depth_sampler=bottle_right)
    assert not any(isinstance(e, KnobRelease) for e in tl.events)
    assert tl.engine.snapshot().state == "engaged"
    tl.step([make_hand(center=RIGHT, pose="release")], n=10, depth_sampler=bottle_right)
    assert any(isinstance(e, KnobRelease) for e in tl.events)


def test_busy_verdict_lingers_past_depth_flicker(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(center=LEFT, pose="pinch")], n=30, depth_sampler=bottle_left)
    # Object gone (or depth flickered): still busy for busy_linger_s (0.5 s).
    tl.step([make_hand(center=LEFT, pose="pinch")], n=10, depth_sampler=flat)
    assert not _engages(tl)
    tl.step([make_hand(center=LEFT, pose="pinch")], n=30, depth_sampler=flat)
    assert len(_engages(tl)) == 1


def test_fist_playpause_blocked_while_holding(cfg):
    """The legacy fist pose has no shape-based object checks of its own —
    a fist wrapped around a bottle relies entirely on the busy-hand gate."""
    cfg.playpause.pose = "fist"
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(center=LEFT, pose="fist")], n=100, depth_sampler=bottle_left)
    assert not any(isinstance(e, PlayPauseHold) for e in tl.events)
    tl.step([make_hand(center=LEFT, pose="fist")], n=60, depth_sampler=flat)
    assert len([e for e in tl.events if isinstance(e, PlayPauseHold)]) == 1


def test_disabled_gate_restores_old_behaviour(cfg):
    cfg.gate.object_gap_m = 0.0
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(center=LEFT, pose="pinch")], n=30, depth_sampler=bottle_left)
    assert len(_engages(tl)) == 1
