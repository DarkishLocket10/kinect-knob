"""Play/pause hold: an open palm FACING the camera, held still, holding nothing.

The guards are the point — a hand waving past, the back of a raised hand, an
edge-on palm, or a hand holding an object must never toggle playback.

Sign convention (field-verified 2026-07-07): conftest's canonical open hand
(make_hand) shows the camera the BACK of the hand; mirroring it about its
palm centre (open_palm below) shows the palm.
"""
from conftest import Timeline, make_hand
from kinectknob.gestures.engine import GestureEngine, finger_spread, palm_facing_score
from kinectknob.types import Hand, PlayPauseHold

# presence gate (0.5 s) + hold (0.7 s) at 30 fps, with slack
HOLD_FRAMES = 45


def _transform_x(hand: Hand, factor: float) -> Hand:
    """Scale the hand's x-coordinates about the palm centre. factor=-1
    mirrors it (back of hand <-> palm); 0 < factor < 1 squeezes it edge-on."""
    pts = hand.pts.copy()
    cx = hand.palm_center[0]
    pts[:, 0] = cx + (pts[:, 0] - cx) * factor
    return Hand(pts=pts, z=hand.z, handedness=hand.handedness, score=hand.score)


def open_palm(center=(320.0, 300.0), angle_deg: float = 0.0) -> Hand:
    """An open hand whose PALM faces the camera."""
    return _transform_x(make_hand(pose="open", center=center, angle_deg=angle_deg), -1.0)


def bunched_palm(center=(320.0, 300.0)) -> Hand:
    """Palm facing the camera, fingers extended — but the fingertips bunched
    together the way they are around a held phone or mug."""
    h = open_palm(center=center)
    pts = h.pts.copy()
    mid = pts[12].copy()                       # middle fingertip
    for tip, dy in ((8, 8.0), (16, 10.0), (20, 14.0)):
        pts[tip][0] = mid[0] + (pts[tip][0] - mid[0]) * 0.15
        pts[tip][1] = mid[1] + dy
    return Hand(pts=pts, z=h.z, handedness=h.handedness, score=h.score)


def _holds(tl):
    return [e for e in tl.events if isinstance(e, PlayPauseHold)]


def test_palm_facing_score_signs():
    back = make_hand(pose="open")
    assert palm_facing_score(back) < -0.5               # canonical hand = back
    assert palm_facing_score(open_palm()) > 0.5         # mirrored = palm
    assert abs(palm_facing_score(_transform_x(back, 0.15))) < 0.3  # edge-on
    # Same geometry with the other handedness label flips the verdict.
    other = Hand(pts=back.pts, z=back.z, handedness="Left", score=back.score)
    assert palm_facing_score(other) > 0.5


def test_finger_spread_discriminates_held_objects():
    assert finger_spread(open_palm()) > 0.3
    assert finger_spread(bunched_palm()) < 0.2


def test_open_palm_hold_toggles_once_then_cooldown(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1
    tl.step([open_palm()], n=30)          # inside the 2 s cooldown
    assert len(_holds(tl)) == 1
    tl.step([open_palm()], n=60)          # cooldown over, still holding
    assert len(_holds(tl)) == 2


def test_back_of_hand_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=100)
    assert _holds(tl) == []


def test_edge_on_palm_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([_transform_x(make_hand(pose="open"), 0.15)], n=100)
    assert _holds(tl) == []


def test_bunched_fingers_never_toggle(cfg):
    """Facing and still — but shaped like a hand pressed around an object."""
    tl = Timeline(GestureEngine(cfg))
    tl.step([bunched_palm()], n=100)
    assert _holds(tl) == []


def test_object_in_front_of_palm_never_toggles(cfg):
    """Depth says the palm surface is ~12 cm nearer than the wrist: that is
    a held object, not an empty palm."""
    # The wrist sits ~75 px below the palm centre in the synthetic hand.
    def object_sampler(x, y):
        return 0.70 if y > 340 else 0.58

    def flat_sampler(x, y):
        return 0.70

    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=100, depth_sampler=object_sampler)
    assert _holds(tl) == []
    tl.step([open_palm()], n=HOLD_FRAMES, depth_sampler=flat_sampler)
    assert len(_holds(tl)) == 1           # empty palm at uniform depth is fine


def test_invert_facing_tunable(cfg):
    cfg.playpause.invert_facing = True
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=100)         # the true palm now reads as back
    assert _holds(tl) == []
    tl.step([make_hand(pose="open")], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1


def test_moving_palm_never_toggles(cfg):
    tl = Timeline(GestureEngine(cfg))
    for i in range(100):                  # facing the camera, but drifting
        tl.step([open_palm(center=(100 + (i % 30) * 15, 300))])
    assert _holds(tl) == []


def test_fist_never_toggles_in_palm_mode(cfg):
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="fist")], n=100)
    assert _holds(tl) == []


def test_hand_entering_frame_does_not_instantly_toggle(cfg):
    cfg.playpause.hold_s = 0.2            # even with a very short hold...
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=10)          # ...the presence gate still blocks
    assert _holds(tl) == []


def test_disabled_toggles_nothing(cfg):
    cfg.playpause.enabled = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=100)
    assert _holds(tl) == []


def test_facing_requirement_can_be_disabled(cfg):
    cfg.playpause.require_facing = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([make_hand(pose="open")], n=HOLD_FRAMES)   # back of hand
    assert len(_holds(tl)) == 1


def test_fist_mode_keeps_old_behaviour(cfg):
    cfg.playpause.pose = "fist"
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm()], n=60)          # open palm: not the pose
    assert _holds(tl) == []
    tl.step([make_hand(pose="fist")], n=HOLD_FRAMES)
    assert len(_holds(tl)) == 1


def test_open_palm_swipe_does_not_end_as_playpause(cfg):
    """In open-palm swipe mode, the palm lingering after the swipe lands
    must not toggle playback — the swipe arms the play/pause cooldown."""
    cfg.swipe.two_finger = False
    tl = Timeline(GestureEngine(cfg))
    tl.step([open_palm(center=(180, 300))], n=15)
    for i in range(1, 7):                 # a real swipe...
        tl.step([open_palm(center=(180 + i * 40, 300))])
    tl.step([open_palm(center=(420, 300))], n=45)      # ...then hold still
    assert _holds(tl) == []               # blocked for the full cooldown


def test_facing_score_survives_rotation(cfg):
    # A facing palm rotated in-plane (tilted hand) still reads as facing:
    # the cross product is rotation-invariant.
    for angle in (-40, -15, 20, 45):
        assert palm_facing_score(open_palm(angle_deg=angle)) > 0.5, angle
