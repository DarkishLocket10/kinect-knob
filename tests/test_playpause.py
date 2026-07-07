"""Play/pause hold: an open palm FACING the camera, held still.

The facing check is the point — a hand waving past, the back of a raised
hand, or an edge-on palm must never toggle playback.
"""
import numpy as np
from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine, palm_facing_score
from kinectknob.types import Hand, PlayPauseHold

# presence gate (0.5 s) + hold (0.7 s) at 30 fps, with slack
HOLD_FRAMES = 45


def _transform_x(hand: Hand, factor: float) -> Hand:
    """Scale the hand's x-coordinates about the palm centre. factor=-1
    mirrors it (shows the back of the hand / anatomically flips it);
    0 < factor < 1 squeezes it toward edge-on."""
    pts = hand.pts.copy()
    cx = hand.palm_center[0]
    pts[:, 0] = cx + (pts[:, 0] - cx) * factor
    return Hand(pts=pts, z=hand.z, handedness=hand.handedness, score=hand.score)


def _holds(tl):
    return [e for e in tl.events if isinstance(e, PlayPauseHold)]


def test_palm_facing_score_signs():
    palm = make_hand(pose="open")
    assert palm_facing_score(palm) > 0.5                       # flat palm, facing
    assert palm_facing_score(_transform_x(palm, -1.0)) < -0.5  # back of hand
    assert abs(palm_facing_score(_transform_x(palm, 0.15))) < 0.3  # edge-on
    # Same geometry with the other handedness label = a left hand showing
    # its back — must read as not-facing.
    flipped_label = Hand(pts=palm.pts, z=palm.z, handedness="Left", score=palm.score)
    assert palm_facing_score(flipped_label) < 0


def test_open_palm_hold_toggles_once_then_cooldown(cfg):
    tl = Timeline(GestureEngine(cfg))
    palm = make_hand(pose="open")
    tl.step([palm], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1
    tl.step([palm], n=30)                 # inside the 2 s cooldown
    assert len(_holds(tl)) == 1
    tl.step([palm], n=60)                 # cooldown over, still holding
    assert len(_holds(tl)) == 2


def test_back_of_hand_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([_transform_x(make_hand(pose="open"), -1.0)], n=100)
    assert _holds(tl) == []


def test_edge_on_palm_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([_transform_x(make_hand(pose="open"), 0.15)], n=100)
    assert _holds(tl) == []


def test_moving_palm_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    for i in range(100):                  # ~0.7 widths/s, facing the camera
        tl.step([make_hand(pose="open", center=(100 + (i % 30) * 15, 300))])
    assert _holds(tl) == []


def test_fist_never_toggles_in_palm_mode(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="fist")], n=100)
    assert _holds(tl) == []


def test_hand_entering_frame_does_not_instantly_toggle(cfg):
    cfg.playpause.hold_s = 0.2            # even with a very short hold...
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=10)   # ...presence gate still blocks
    assert _holds(tl) == []


def test_disabled_toggles_nothing(cfg):
    cfg.playpause.enabled = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=100)
    assert _holds(tl) == []


def test_facing_requirement_can_be_disabled(cfg):
    cfg.playpause.require_facing = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([_transform_x(make_hand(pose="open"), -1.0)], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1


def test_fist_mode_keeps_old_behaviour(cfg):
    cfg.playpause.pose = "fist"
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=60)   # open palm: not the pose
    assert _holds(tl) == []
    tl.step([make_hand(pose="fist")], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1


def test_open_palm_swipe_does_not_end_as_playpause(cfg):
    """In open-palm swipe mode, the palm lingering after the swipe lands
    must not toggle playback — the swipe arms the play/pause cooldown."""
    cfg.swipe.two_finger = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open", center=(180, 300))], n=15)
    for i in range(1, 7):                 # a real swipe...
        tl.step([make_hand(pose="open", center=(180 + i * 40, 300))])
    tl.step([make_hand(pose="open", center=(420, 300))], n=45)  # ...then hold
    assert _holds(tl) == []               # blocked for the full cooldown


def test_facing_score_survives_rotation(cfg):
    # A facing palm rotated in-plane (tilted hand) still reads as facing:
    # the cross product is rotation-invariant.
    for angle in (-40, -15, 20, 45):
        h = make_hand(pose="open", angle_deg=angle)
        assert palm_facing_score(h) > 0.5, angle
