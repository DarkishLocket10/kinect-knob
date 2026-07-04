"""Signal filtering: angle wrapping and the One Euro filter.

The One Euro filter (Casiez et al., CHI 2012) is the standard choice for
low-latency interactive smoothing: heavy smoothing when the signal is slow
(kills jitter), light smoothing when it moves fast (no perceptible lag).
"""
from __future__ import annotations

import math


def wrap_deg(a: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


class _LowPass:
    def __init__(self) -> None:
        self.initialized = False
        self.y = 0.0

    def filter(self, x: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """One Euro filter over an irregularly-sampled scalar signal.

    min_cutoff: baseline cutoff (Hz). Lower = smoother but laggier at rest.
    beta: speed coefficient. Higher = snappier during fast motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._dx.filter(0.0, 1.0)
            return self._x.filter(x, 1.0)
        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3
        self._t_prev = t
        dx = (x - self._x.y) / dt
        edx = self._dx.filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(x, self._alpha(cutoff, dt))
