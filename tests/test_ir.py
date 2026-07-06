"""Kinect v2 IR night mode: auto-switch hysteresis and tone mapping."""
from __future__ import annotations

import numpy as np
import pytest

from kinectknob.capture.ir import (
    BRIGHT_LUMA,
    DARK_LUMA,
    DWELL_FRAMES,
    IrAutoSwitch,
    ir_to_rgb,
)

DARK = DARK_LUMA - 10
BRIGHT = BRIGHT_LUMA + 10
DUSK = (DARK_LUMA + BRIGHT_LUMA) / 2  # inside the hysteresis band


def feed(sw: IrAutoSwitch, luma: float, n: int) -> bool:
    out = sw.active
    for _ in range(n):
        out = sw.update(luma)
    return out


class TestIrAutoSwitch:
    def test_off_never_activates(self):
        sw = IrAutoSwitch("off")
        assert not feed(sw, 0.0, DWELL_FRAMES * 3)

    def test_always_is_always_on(self):
        sw = IrAutoSwitch("always")
        assert sw.active
        assert feed(sw, 255.0, DWELL_FRAMES * 3)

    def test_auto_starts_on_color(self):
        assert not IrAutoSwitch("auto").active

    def test_switches_to_ir_after_sustained_darkness(self):
        sw = IrAutoSwitch("auto")
        assert not feed(sw, DARK, DWELL_FRAMES - 1)
        assert sw.update(DARK)

    def test_switches_back_after_sustained_brightness(self):
        sw = IrAutoSwitch("auto")
        feed(sw, DARK, DWELL_FRAMES)
        assert sw.active
        assert feed(sw, BRIGHT, DWELL_FRAMES - 1)
        assert not sw.update(BRIGHT)

    def test_dusk_luma_never_flips_either_way(self):
        sw = IrAutoSwitch("auto")
        assert not feed(sw, DUSK, DWELL_FRAMES * 3)
        feed(sw, DARK, DWELL_FRAMES)
        assert feed(sw, DUSK, DWELL_FRAMES * 3)  # stays in IR at dusk too

    def test_flicker_resets_the_dwell_counter(self):
        sw = IrAutoSwitch("auto")
        for _ in range(5):
            feed(sw, DARK, DWELL_FRAMES - 1)
            sw.update(BRIGHT)  # a single bright frame (TV flash) resets
        assert not sw.active

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            IrAutoSwitch("night")


class TestIrToRgb:
    def test_shape_dtype_and_channel_identity(self):
        ir = np.random.default_rng(0).uniform(0, 65535, (424, 512)).astype(np.float32)
        rgb = ir_to_rgb(ir)
        assert rgb.shape == (424, 512, 3)
        assert rgb.dtype == np.uint8
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 0], rgb[..., 2])

    def test_full_range_maps_to_full_range(self):
        ir = np.array([[0.0, 65535.0]], dtype=np.float32)
        rgb = ir_to_rgb(ir)
        assert rgb[0, 0, 0] == 0
        assert rgb[0, 1, 0] == 255

    def test_sqrt_lifts_the_dim_midrange(self):
        # A hand at 2 m reflects a weak signal; linear mapping would leave it
        # nearly black. 10% signal must land well above 10% brightness.
        rgb = ir_to_rgb(np.full((2, 2), 6553.5, dtype=np.float32))
        assert rgb[0, 0, 0] > 60

    def test_squeezes_trailing_axis_and_clips_outliers(self):
        ir = np.full((4, 4, 1), 1e9, dtype=np.float32)  # over-range hotspot
        rgb = ir_to_rgb(ir)
        assert rgb.shape == (4, 4, 3)
        assert rgb.max() == 255
