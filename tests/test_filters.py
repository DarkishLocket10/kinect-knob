import math

from kinectknob.filters import OneEuroFilter, wrap_deg


def test_wrap_deg():
    assert wrap_deg(0) == 0
    assert wrap_deg(179) == 179
    assert wrap_deg(180) == -180
    assert wrap_deg(-181) == 179
    assert wrap_deg(360) == 0
    assert wrap_deg(365) == 5
    assert wrap_deg(-365) == -5


def test_one_euro_converges_at_rest():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    y = 0.0
    for _ in range(120):  # 4 seconds at 30 fps of a constant signal
        t += 1 / 30
        y = f(t, 50.0)
    assert abs(y - 50.0) < 0.5


def test_one_euro_smooths_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    t = 0.0
    outputs = []
    for i in range(90):
        t += 1 / 30
        noisy = 10.0 + (2.0 if i % 2 == 0 else -2.0)  # +/-2 alternating jitter
        outputs.append(f(t, noisy))
    tail = outputs[30:]
    # Filtered signal should wobble far less than the +/-2 input jitter.
    assert max(tail) - min(tail) < 1.0


def test_one_euro_tracks_fast_motion_with_low_lag():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t = 0.0
    x = 0.0
    y = 0.0
    for _ in range(30):  # 1 second sweep to 300
        t += 1 / 30
        x += 10.0
        y = f(t, x)
    assert x - y < 40.0  # lag stays small during fast motion


def test_one_euro_reset():
    f = OneEuroFilter()
    f(0.1, 100.0)
    f.reset()
    assert f(0.2, 5.0) == 5.0


def test_one_euro_handles_nonincreasing_time():
    f = OneEuroFilter()
    f(1.0, 1.0)
    y = f(1.0, 2.0)  # dt == 0 must not divide by zero
    assert math.isfinite(y)
