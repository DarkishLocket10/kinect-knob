"""Low-light auto-gamma boost + exposure spec parsing."""
import numpy as np
import pytest

from kinectknob.capture.lowlight import LowLightBoost
from kinectknob.config import parse_exposure


def _frame(luma: int) -> np.ndarray:
    return np.full((120, 160, 3), luma, dtype=np.uint8)


def test_bright_frames_pass_through_untouched():
    boost = LowLightBoost()
    frame = _frame(140)
    for _ in range(30):
        out = boost.process(frame)
    assert out is frame                       # identity, not even a copy
    assert not boost.active


def test_dim_frames_are_lifted_toward_target():
    boost = LowLightBoost()
    frame = _frame(35)
    for _ in range(60):                       # let the EMA converge
        out = boost.process(frame)
    assert boost.active
    assert 90 <= float(out.mean()) <= 130     # near TARGET_LUMA
    assert out.dtype == np.uint8


def test_black_frame_lift_is_capped():
    boost = LowLightBoost()
    for _ in range(60):
        out = boost.process(_frame(0))
    assert float(out.mean()) == 0.0           # gamma can't invent photons
    # ...and the exponent hit its floor rather than exploding.
    assert boost._exponent == pytest.approx(0.45, abs=0.02)


def test_boost_adapts_smoothly_not_instantly():
    """One bright TV flash mid-darkness must not strobe the brightness."""
    boost = LowLightBoost()
    for _ in range(60):
        boost.process(_frame(35))
    dark_exponent = boost._exponent
    boost.process(_frame(200))                # single bright flash
    assert abs(boost._exponent - dark_exponent) < 0.15


def test_parse_exposure_specs():
    assert parse_exposure("auto") == ("auto", ())
    assert parse_exposure("auto:1.5") == ("auto", (1.5,))
    assert parse_exposure("semi:8") == ("semi", (8.0,))
    assert parse_exposure("SEMI:8.5 ") == ("semi", (8.5,))
    assert parse_exposure("manual:8,2") == ("manual", (8.0, 2.0))
    # Clamped to the sensor's ranges.
    assert parse_exposure("semi:500") == ("semi", (66.0,))
    assert parse_exposure("manual:0.01,9") == ("manual", (0.1, 4.0))
    for bad in ("", "fast", "semi:", "semi:abc", "manual:8", "manual:,2"):
        with pytest.raises(ValueError):
            parse_exposure(bad)
