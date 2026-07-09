"""Crop-in: zoom the tracked image onto the centre of the frame.

The Kinect sees a wide slice of the room; the person controlling it usually
occupies the middle. Cropping before tracking (a) makes the hand a larger
fraction of what the landmark model sees — better detection at distance —
and (b) removes the space around the user entirely, so motion at the frame
edges (doorways, TV, other people) can't even become a candidate hand.

The SAME zoom must be applied to the rgb frame and its aligned depth map:
the engine's depth sampler maps tracked pixel coordinates into the depth
array by relative scale, which survives any equal-fraction crop.

Pure numpy; returns views (no copy).
"""
from __future__ import annotations

import numpy as np

_MIN_SIDE = 32  # never crop below this many pixels per side


def center_crop(arr: np.ndarray, zoom: float) -> np.ndarray:
    """The central 1/zoom window of an HxW or HxWxC array. zoom <= 1 is the
    identity. Returns a view into the input."""
    if zoom <= 1.0:
        return arr
    h, w = arr.shape[:2]
    cw = max(int(round(w / zoom)), min(_MIN_SIDE, w))
    ch = max(int(round(h / zoom)), min(_MIN_SIDE, h))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]
