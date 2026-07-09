"""Swipe gesture detection (two-finger pose by default; open-palm legacy mode)."""
from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine, openness
from kinectknob.types import KnobEngage, Swipe


def _settle(tl, x=320.0, pose="two"):
    """Hand present and still long enough to pass the presence gate."""
    tl.step([make_hand(pose=pose, center=(x, 300))], n=15)  # 0.5 s


def _swipe(tl, start_x, step, pose="two"):
    for i in range(1, 7):  # 240 px in 0.2 s
        tl.step([make_hand(pose=pose, center=(start_x + i * step, 300))])


def test_two_finger_pose_classifier():
    assert openness(make_hand(pose="two")) == "two"
    assert openness(make_hand(pose="open")) == "open"


def test_swipe_right_is_next(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    _swipe(tl, 180, 40)
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == 1


def test_swipe_left_is_previous(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=460)
    _swipe(tl, 460, -40)
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == -1


def test_open_palm_does_not_swipe_in_two_finger_mode(cfg):
    # The point of the new pose: waving an open hand across the frame — the
    # classic accidental skip — no longer registers.
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180, pose="open")
    _swipe(tl, 180, 40, pose="open")
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_open_palm_mode_still_works(cfg):
    cfg.swipe.two_finger = False
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180, pose="open")
    _swipe(tl, 180, 40, pose="open")
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == 1


def test_two_finger_swipe_tolerates_blur_frames(cfg):
    # Fast lateral motion blurs landmarks and the pose misclassifies for a
    # frame or two; the generous 75% match must ride through it.
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    for i in range(1, 7):
        pose = "open" if i == 3 else "two"   # frame 3: mid-swipe misread
        tl.step([make_hand(pose=pose, center=(180 + i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1


def test_swipe_survives_blur_dropout_frames(cfg):
    """A fast swipe blurs the hand enough for tracking to LOSE it entirely
    for a frame or two mid-motion. The lost-grace must carry the swipe's
    history and presence through the gap — this used to kill every fast
    swipe (history wiped + presence clock reset by a single empty frame)."""
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    for i in range(1, 7):
        if i in (3, 4):                       # blur: hand vanishes mid-swipe
            tl.step([])
        else:
            tl.step([make_hand(pose="two", center=(180 + i * 40, 300))])
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == 1


def test_long_dropout_still_resets_presence(cfg):
    """The grace is for blur flickers, not for leaving: after a gap longer
    than gate.lost_grace_s the presence gate applies afresh, so a hand
    re-entering the frame mid-motion can't skip a track."""
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    tl.step([], n=12)                          # 0.4 s > lost_grace_s (0.25)
    _swipe(tl, 180, 40)                        # immediately swipes on return
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_swipe_invert_flips_direction(cfg):
    cfg.swipe.invert = True
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=180)
    _swipe(tl, 180, 40)
    swipes = [e for e in tl.events if isinstance(e, Swipe)]
    assert len(swipes) == 1
    assert swipes[0].direction == -1   # rightward swipe reads as previous


def test_slow_drift_does_not_swipe(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=200)
    for i in range(1, 40):  # 200 px over 1.3 s — too slow
        tl.step([make_hand(pose="two", center=(200 + i * 5, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_vertical_wave_does_not_swipe(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl)
    for i in range(1, 7):
        tl.step([make_hand(pose="two", center=(320 + i * 12, 300 - i * 40))])
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
        tl.step([make_hand(pose="two", center=(20 + i * 45, 300))])
    assert not any(isinstance(e, Swipe) for e in tl.events)


def test_swipe_cooldown(cfg):
    tl = Timeline(GestureEngine(cfg))
    _settle(tl, x=140)
    _swipe(tl, 140, 40)
    # Immediately swipe back the other way — inside the cooldown.
    _swipe(tl, 380, -40)
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
