"""Crop-in zoom: the tracked view narrows to the frame centre, and rgb +
depth must crop by the same fraction so the depth sampler stays aligned."""
import numpy as np

from kinectknob.capture.crop import center_crop


def test_zoom_one_is_identity():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert center_crop(rgb, 1.0) is rgb
    assert center_crop(rgb, 0.5) is rgb       # nonsense zoom-out: identity


def test_crop_keeps_the_centre():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[240, 320] = 255                        # centre pixel
    rgb[0, 0] = 128                            # corner pixel
    out = center_crop(rgb, 2.0)
    assert out.shape == (240, 320, 3)
    assert out[120, 160, 0] == 255             # centre stays centred
    assert not (out == 128).any()              # the corner is gone


def test_rgb_and_depth_crop_stay_aligned():
    """Same zoom on differently-sized aligned arrays must keep relative
    coordinates identical — that's what the depth sampler relies on."""
    rgb = np.zeros((360, 480, 3), dtype=np.uint8)
    depth = np.zeros((424, 512), dtype=np.float32)
    zoom = 1.6
    rgb_c, depth_c = center_crop(rgb, zoom), center_crop(depth, zoom)
    assert abs(rgb_c.shape[1] / rgb.shape[1] - depth_c.shape[1] / depth.shape[1]) < 0.01
    assert abs(rgb_c.shape[0] / rgb.shape[0] - depth_c.shape[0] / depth.shape[0]) < 0.01


def test_returns_view_not_copy():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = center_crop(rgb, 1.5)
    out[0, 0] = 7
    # crop window is 427x320 at origin (x0=106, y0=80)
    assert rgb[80, 106, 0] == 7                # writes land in the parent


def test_extreme_zoom_clamped_to_min_side():
    tiny = center_crop(np.zeros((480, 640), dtype=np.uint8), 100.0)
    assert min(tiny.shape) >= 32
