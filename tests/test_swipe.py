"""Swipe gesture detection."""
from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine
from kinectknob.types import KnobEngage, Swipe


def _settle(tl, x=320.0):
    """Open hand present and still long enough to pass the presence gate."""
    tl.step([make_hand(pose="open", center=(x, 300))], n=15)  # 0.5 s


def test_swipe_right_is_next(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    for i in range(1, 7):  # 240 px in 0.2 s
        tl.step([make_hand(pose="open", center=(180 + i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == 1


def test_swipe_left_is_previous(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=460)
    for i in range(1, 7):
        tl.step([make_hand(pose="open", center=(460 - i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == -1


def test_swipe_invert_flips_direction(cfg):
    cfg.swipe.invert = True
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    for i in range(1, 7):  # rightward swipe...
        tl.step([make_hand(pose="open", center=(180 + i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == -1   # ...reads as previous when inverted


def test_slow_drift_does_not_swipe(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=200)
    for i in range(1, 40):  # 200 px over 1.3 s — too slow
        tl.step([make_hand(pose="open", center=(200 + i * 5, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_vertical_wave_does_not_swipe(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl)
    for i in range(1, 7):
        tl.step([make_hand(pose="open", center=(320 + i * 12, 300 - i * 40))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_fist_movement_does_not_swipe(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="fist", center=(180, 300))], n=15)
    for i in range(1, 7):
        tl.step([make_hand(pose="fist", center=(180 + i * 40, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_hand_entering_frame_does_not_swipe(cfg):
    """A hand sweeping in from the edge must not skip a track."""
    tl = Timeline(GestureEngine(cfg))
    for i in range(8):  # appears at the edge already moving
        tl.step([make_hand(pose="open", center=(20 + i * 45, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_swipe_cooldown(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=140)
    for i in range(1, 7):
        tl.step([make_hand(pose="open", center=(140 + i * 40, 300))])
    # Immediately swipe back the other way — inside the cooldown.
    for i in range(1, 7):
        tl.step([make_hand(pose="open", center=(380 - i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1


def test_no_swipe_while_knob_engaged(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="pinch")], n=6)
    assert any(isinstance(e, KnobEngage) for e in tl.events)
    # Drag the pinched hand quickly: knob is engaged, so no swipe.
    for i in range(1, 7):
        tl.step([make_hand(pose="pinch", center=(320 + i * 40, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)
