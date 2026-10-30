"""Crop-in zoom and pan: the tracked view narrows to part of the frame, and
rgb + depth must take the same window — same fraction AND same offset — so the
depth sampler stays aligned."""
import numpy as np

from kinectknob.capture.crop import _origin, crop_view


def test_zoom_one_is_identity():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert crop_view(rgb, 1.0) is rgb
    assert crop_view(rgb, 0.5) is rgb       # nonsense zoom-out: identity


def test_crop_keeps_the_centre():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[240, 320] = 255                        # centre pixel
    rgb[0, 0] = 128                            # corner pixel
    out = crop_view(rgb, 2.0)
    assert out.shape == (240, 320, 3)
    assert out[120, 160, 0] == 255             # centre stays centred
    assert not (out == 128).any()              # the corner is gone


def test_rgb_and_depth_crop_stay_aligned():
    """Same zoom on differently-sized aligned arrays must keep relative
    coordinates identical — that's what the depth sampler relies on."""
    rgb = np.zeros((360, 480, 3), dtype=np.uint8)
    depth = np.zeros((424, 512), dtype=np.float32)
    zoom = 1.6
    rgb_c, depth_c = crop_view(rgb, zoom), crop_view(depth, zoom)
    assert abs(rgb_c.shape[1] / rgb.shape[1] - depth_c.shape[1] / depth.shape[1]) < 0.01
    assert abs(rgb_c.shape[0] / rgb.shape[0] - depth_c.shape[0] / depth.shape[0]) < 0.01


def test_returns_view_not_copy():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = crop_view(rgb, 1.5)
    out[0, 0] = 7
    # crop window is 427x320 at origin (x0=106, y0=80)
    assert rgb[80, 106, 0] == 7                # writes land in the parent


def test_extreme_zoom_clamped_to_min_side():
    tiny = crop_view(np.zeros((480, 640), dtype=np.uint8), 100.0)
    assert min(tiny.shape) >= 32


# ----------------------------------------------------------------- panning


def test_pan_zero_is_the_old_centred_crop():
    """Existing setups must not shift by a pixel now that pan exists."""
    for full, side in ((640, 427), (480, 320), (512, 313), (101, 50)):
        assert _origin(full, side, 0.0) == (full - side) // 2


def test_pan_extremes_sit_flush_against_the_edges():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, 0] = 11                             # leftmost column
    rgb[:, -1] = 22                            # rightmost column
    rgb[0, :] = 33                             # top row
    rgb[-1, :] = 44                            # bottom row

    left = crop_view(rgb, 2.0, pan_x=-1.0)
    assert (left[:, 0] == 11).all() and not (left == 22).any()

    right = crop_view(rgb, 2.0, pan_x=1.0)
    assert (right[:, -1] == 22).all() and not (right == 11).any()

    top = crop_view(rgb, 2.0, pan_y=-1.0)
    assert (top[0] == 33).all() and not (top == 44).any()

    bottom = crop_view(rgb, 2.0, pan_y=1.0)
    assert (bottom[-1] == 44).all() and not (bottom == 33).any()


def test_pan_never_leaves_the_frame_however_hard_it_is_pushed():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    for pan in (-9.0, -1.0, 0.0, 1.0, 9.0):
        out = crop_view(rgb, 2.5, pan_x=pan, pan_y=pan)
        assert out.shape[0] > 0 and out.shape[1] > 0
        assert out.base is rgb or out.base is rgb.base   # still a view, in bounds
    # clamping is on the pan, not the arithmetic: beyond the ends is the end
    assert _origin(640, 256, -9.0) == _origin(640, 256, -1.0) == 0
    assert _origin(640, 256, 9.0) == _origin(640, 256, 1.0) == 640 - 256


def test_pan_moves_the_window_monotonically():
    origins = [_origin(640, 256, p) for p in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert origins == sorted(origins)
    assert origins[0] == 0 and origins[-1] == 384
    assert origins[2] == 192                    # centred


def test_pan_is_inert_without_zoom():
    """Nothing to slide within a full-frame window, at any pan."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert crop_view(rgb, 1.0, pan_x=1.0, pan_y=-1.0) is rgb


def test_rgb_and_depth_stay_aligned_under_pan():
    """The invariant that matters: differently-sized aligned arrays given the
    same pan must put their windows at the same RELATIVE offset, or every
    depth lookup for a tracked pixel lands in the wrong place."""
    rgb = np.zeros((360, 480, 3), dtype=np.uint8)
    depth = np.zeros((424, 512), dtype=np.float32)
    zoom = 1.6
    for pan in (-1.0, -0.4, 0.0, 0.25, 1.0):
        r = crop_view(rgb, zoom, pan_x=pan, pan_y=pan)
        d = crop_view(depth, zoom, pan_x=pan, pan_y=pan)
        # offset of each window as a fraction of its own frame
        r_off_x = _origin(rgb.shape[1], r.shape[1], pan) / rgb.shape[1]
        d_off_x = _origin(depth.shape[1], d.shape[1], pan) / depth.shape[1]
        r_off_y = _origin(rgb.shape[0], r.shape[0], pan) / rgb.shape[0]
        d_off_y = _origin(depth.shape[0], d.shape[0], pan) / depth.shape[0]
        assert abs(r_off_x - d_off_x) < 0.01, pan
        assert abs(r_off_y - d_off_y) < 0.01, pan
