"""Kinect v2 color exposure control via the libkk_exposure.so bridge.

The pinned freenect2 python binding (0.2.3) holds the raw C++
``Freenect2Device*`` in ``Device._c_object`` but never wrapped libfreenect2's
exposure API. ``native/kk_exposure.cpp`` (compiled into the Docker image as
``libkk_exposure.so``) bridges the gap: it takes that same pointer and calls
``setColor{Auto,SemiAuto,Manual}Exposure``.

Why this matters: in dim rooms the color camera's auto-exposure stretches the
integration time toward ~33 ms and halves the stream to 15 fps — fast hands
smear into a blur the landmark model can't track, which is what kills swipes.
``semi:<ms>`` caps the integration time (shutter) while analog gain floats,
trading blur for noise — the right trade for tracking. The resulting darker
frames are compensated in software by the low-light boost (lowlight.py).
"""
from __future__ import annotations

import ctypes
import logging

log = logging.getLogger("kk.cap.exposure")

_LIB_NAME = "libkk_exposure.so"


def _device_ptr(device) -> int:
    """The binding's Device._c_object is a cffi void* holding the raw
    Freenect2Device* — turn it into an int for ctypes."""
    import freenect2

    return int(freenect2.ffi.cast("uintptr_t", device._c_object))


def apply_exposure(device, spec: str) -> bool:
    """Apply a parsed capture.exposure spec to a started freenect2 Device.
    Returns True when a setting was sent, False for plain "auto" (nothing to
    do). Raises OSError if the bridge library is missing (non-Docker runs)."""
    from ..config import parse_exposure

    mode, args = parse_exposure(spec)
    if mode == "auto" and not args:
        return False

    lib = ctypes.CDLL(_LIB_NAME)
    for fn in (
        lib.kk_set_color_auto_exposure,
        lib.kk_set_color_semi_auto_exposure,
        lib.kk_set_color_manual_exposure,
    ):
        fn.restype = ctypes.c_int
    lib.kk_set_color_auto_exposure.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.kk_set_color_semi_auto_exposure.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.kk_set_color_manual_exposure.argtypes = [
        ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
    ]

    ptr = ctypes.c_void_p(_device_ptr(device))
    if mode == "auto":
        rc = lib.kk_set_color_auto_exposure(ptr, args[0])
    elif mode == "semi":
        rc = lib.kk_set_color_semi_auto_exposure(ptr, args[0])
    else:
        rc = lib.kk_set_color_manual_exposure(ptr, args[0], args[1])
    if rc != 0:
        raise RuntimeError(f"exposure bridge returned {rc} for {spec!r}")
    return True
