"""Presence detection: the empty-room gate and the per-region occupancy signal.

The scenes here are synthetic depth maps in millimetres, built the way the
Kinect delivers them: a flat-ish background plane with objects standing in
front of it, and 0 where the sensor got no return.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinectknob.presence import DepthPresence, RegionPresence


def wall(h=120, w=160, mm=2500.0, noise=0.0, seed=0) -> np.ndarray:
    """An empty scene: a wall at a constant distance, optionally noisy."""
    arr = np.full((h, w), mm, dtype=np.float32)
    if noise:
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape).astype(np.float32)
    return arr


def with_body(base: np.ndarray, frac: float = 0.25, mm: float = 1500.0) -> np.ndarray:
    """Put a person-shaped slab in front of the wall, covering ``frac`` of it."""
    out = base.copy()
    h, w = out.shape
    rows = max(1, int(round(h * frac)))
    out[h - rows:, : w] = mm
    return out


def settle(det: DepthPresence, frame: np.ndarray, t0: float, n: int, step: float = 0.5):
    """Run n probes, returning the last result and the time reached."""
    res = None
    t = t0
    for _ in range(n):
        res = det.update(frame, t)
        t += step
    return res, t


# -- whole-scene presence ---------------------------------------------


def test_empty_room_reads_absent():
    det = DepthPresence(warmup_s=0.0)
    res, _ = settle(det, wall(), 0.0, 6)
    assert res.occupancy == 0.0
    assert det.present is False


def test_person_wakes_the_pipeline():
    det = DepthPresence(warmup_s=0.0, linger_s=5.0)
    _, t = settle(det, wall(), 0.0, 6)
    res = det.update(with_body(wall()), t)
    assert res.occupancy > 0.2
    assert det.present is True
    assert res.nearest_mm == 1500


def test_a_head_sized_intrusion_is_enough():
    """The whole point of occupancy over percentiles: a small intrusion — a
    head leaning in — must register, not be averaged away."""
    det = DepthPresence(warmup_s=0.0, on_frac=0.02)
    _, t = settle(det, wall(), 0.0, 6)
    scene = wall()
    scene[10:22, 30:52] = 1400.0          # ~1.4% of the frame... just under
    res = det.update(scene, t)
    assert 0.005 < res.occupancy < 0.02
    assert det.present is False           # honest: below the configured gate
    scene[10:28, 30:60] = 1400.0          # ~2.8% — a head at closer range
    res = det.update(scene, t + 0.5)
    assert res.occupancy > 0.02
    assert det.present is True


def test_presence_lingers_after_you_step_out():
    """Somebody who ducked out of frame has not left the room, and a photo
    taken in that gap is exactly the one that catches half a person."""
    det = DepthPresence(warmup_s=0.0, linger_s=20.0)
    _, t = settle(det, wall(), 0.0, 6)
    det.update(with_body(wall()), t)
    assert det.present is True
    res = det.update(wall(), t + 5.0)     # gone, but only for 5 s
    assert res.occupancy == pytest.approx(0.0, abs=1e-6)
    assert det.present is True
    det.update(wall(), t + 25.0)          # past the linger
    assert det.present is False


def test_a_sitting_person_is_never_absorbed_into_the_background():
    """Yash sits nearly still at the work board for hours. A background model
    that kept adapting would quietly decide he is furniture — which is the
    failure that lets a scan photograph a board he is sitting in front of.

    "Nearly still" is the point: a person shifts a few centimetres now and
    then, and that is exactly what the absorber watches for.
    """
    det = DepthPresence(warmup_s=0.0, linger_s=5.0, static_absorb_s=60.0)
    _, t = settle(det, wall(), 0.0, 6)
    rng = np.random.default_rng(1)
    res = None
    for i in range(400):                 # ~3.5 simulated minutes of sitting
        body = with_body(wall(), mm=1500.0 + rng.normal(0, 80))
        res = det.update(body, t)
        t += 0.5
    assert res.occupancy > 0.2
    assert det.present is True


def test_perfectly_static_furniture_is_eventually_absorbed():
    """A chair somebody parked in view is 'present' at first. If it never
    moves again it has to become part of the room, or the pipeline never
    sleeps and every whiteboard scan is blocked forever."""
    det = DepthPresence(warmup_s=0.0, linger_s=1.0, on_frac=0.02,
                        off_frac=0.01, static_absorb_s=30.0)
    _, t = settle(det, wall(), 0.0, 6)
    chair = wall()
    chair[100:, 0:40] = 1800.0
    det.update(chair, t)
    assert det.present is True
    res, t = settle(det, chair, t + 2.0, 600)
    assert det.present is False
    assert res.occupancy < 0.01


def test_absorption_waits_out_the_static_window():
    """Nothing is absorbed early: the clock only starts once motion stops."""
    det = DepthPresence(warmup_s=0.0, linger_s=1.0, static_absorb_s=300.0)
    _, t = settle(det, wall(), 0.0, 6)
    chair = wall()
    chair[100:, 0:40] = 1800.0
    res, t = settle(det, chair, t, 200)    # 100 s — well inside the window
    assert res.occupancy > 0.02


def test_no_depth_is_not_an_empty_room():
    det = DepthPresence(warmup_s=0.0)
    res = det.update(None, 0.0)
    assert res.present is False
    assert det.snapshot()["ready"] is False


def test_warmup_suppresses_the_seeding_frame():
    """The model seeds from whatever it first sees; verdicts during the warmup
    would be about the seeding, not about the room."""
    det = DepthPresence(warmup_s=3.0)
    det.update(with_body(wall()), 0.0)
    assert det.present is False
    det.update(with_body(wall(), frac=0.5), 0.5)
    assert det.present is False


def test_force_relearn_forgets_the_scene():
    det = DepthPresence(warmup_s=0.0)
    _, t = settle(det, wall(), 0.0, 6)
    det.update(with_body(wall()), t)
    assert det.present is True
    det.force_relearn()
    assert det.snapshot()["ready"] is False
    assert det.present is False


def test_snapshot_reports_what_a_consumer_needs():
    det = DepthPresence(warmup_s=0.0, linger_s=30.0)
    _, t = settle(det, wall(), 0.0, 6)
    det.update(with_body(wall()), t)
    snap = det.snapshot()
    assert snap["present"] is True
    assert snap["ready"] is True
    assert snap["linger_s"] == 30.0
    assert snap["since_seen_s"] is not None
    assert snap["occupancy"] > 0.2


# -- per-region occupancy (the whiteboard signal) ----------------------


def board_scene(h=1080, w=1920, plane=1950.0) -> np.ndarray:
    return np.full((h, w), plane, dtype=np.float32)


def test_region_occupancy_sees_a_head_a_percentile_guard_misses():
    """The regression this whole feature exists for. A head in front of one
    board half covers a few percent of it: p10-vs-p90 (the old region_depth
    guard) does not move, occupancy does."""
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=0.0)
    scene = board_scene()
    rp.update(scene)
    region = (100, 200, 900, 800)       # left board half, unmirrored coords
    clear = rp.occupancy(*region)
    assert clear["occupancy"] == 0.0

    head = scene.copy()
    # ~120x160 px of head inside an 800x600 region = ~4% coverage.
    head[300:460, 1920 - 500:1920 - 380] = 1000.0
    rp.update(head)
    blocked = rp.occupancy(*region)
    assert blocked["occupancy"] > 0.02
    assert blocked["nearest_mm"] == 1000

    # What the percentile test would have concluded from the same region.
    sub = head[200:800, 1920 - 900:1920 - 100]
    p10, p90 = np.percentile(sub, [10, 90])
    assert p10 > p90 - 400          # i.e. depth_blocked() says "clear"


def test_region_occupancy_is_per_region():
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=0.0)
    scene = board_scene()
    rp.update(scene)
    blocked = scene.copy()
    blocked[300:700, 1920 - 900:1920 - 500] = 1200.0   # left half only
    rp.update(blocked)
    left = rp.occupancy(100, 200, 960, 800)
    right = rp.occupancy(960, 200, 1820, 800)
    assert left["occupancy"] > 0.05
    assert right["occupancy"] == 0.0


def test_region_plane_is_learned_not_assumed():
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=0.0)
    rp.update(board_scene(plane=3000.0))
    out = rp.occupancy(100, 200, 900, 800)
    assert out["plane_mm"] == 3000


def test_region_plane_does_not_grow_around_someone_working_there():
    """The whiteboard failure this guards: he leans at the board for an hour,
    the plane creeps out to meet him, and the scan decides the board is
    clear and photographs him."""
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=0.0, static_absorb_s=60.0)
    rp.update(board_scene(), now=0.0)
    rng = np.random.default_rng(2)
    region = (100, 200, 960, 800)
    t = 1.0
    out = None
    for _ in range(300):                   # 5 simulated minutes at the board
        scene = board_scene()
        scene[300:700, 1920 - 900:1920 - 500] = 1200.0 + rng.normal(0, 90)
        rp.update(scene, now=t)
        out = rp.occupancy(*region)
        t += 1.0
    assert out["occupancy"] > 0.05


def test_region_updates_are_rate_limited():
    """The aligned map is 1920 wide and registration runs up to 15x a second;
    the board plane does not need that."""
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=1.0)
    rp.update(board_scene(), now=0.0)
    blocked = board_scene()
    blocked[300:700, 500:900] = 1200.0
    rp.update(blocked, now=0.1)          # inside the rate limit: ignored
    assert rp.occupancy(100, 200, 900, 800)["occupancy"] == 0.0
    rp.update(blocked, now=1.5)          # past it: taken
    assert rp.occupancy(1920 - 900, 200, 1920 - 500, 800)["occupancy"] > 0.5


def test_region_query_before_any_depth_returns_nothing():
    rp = RegionPresence()
    assert rp.occupancy(0, 0, 100, 100) is None


def test_region_coordinates_are_clamped_and_ordered():
    rp = RegionPresence(fg_gap_mm=250.0, min_interval_s=0.0)
    rp.update(board_scene())
    assert rp.occupancy(900, 800, 100, 200) is not None   # reversed
    assert rp.occupancy(-50, -50, 99999, 99999) is not None
    assert rp.occupancy(100, 200, 100, 800) is None       # zero width


def test_dashboard_tuning_applies_without_a_restart():
    """Every other tunable in this app is read fresh per frame. A presence
    detector holding construction-time copies would show sliders that do
    nothing at all — worse than not offering them."""
    from kinectknob.config import PresenceConfig

    det = DepthPresence(warmup_s=0.0, on_frac=0.5, linger_s=1.0)
    _, t = settle(det, wall(), 0.0, 6)
    body = with_body(wall(), frac=0.25)
    det.update(body, t)
    assert det.present is False          # 25% occupancy, 50% threshold

    cfg = PresenceConfig(on_frac=0.02, off_frac=0.01, linger_s=1.0,
                         warmup_s=0.0, fg_gap_m=0.25, near_m=0.4, far_m=5.0)
    det.configure(cfg)
    det.update(body, t + 0.5)
    assert det.present is True
    assert det.snapshot()["on_frac"] == 0.02
