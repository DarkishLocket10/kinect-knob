"""Crop-in: zoom the tracked image onto part of the frame, and pan that window.

The Kinect sees a wide slice of the room; the person controlling it usually
occupies the middle. Cropping before tracking (a) makes the hand a larger
fraction of what the landmark model sees — better detection at distance —
and (b) removes the space around the user entirely, so motion at the frame
edges (doorways, TV, other people) can't even become a candidate hand.

The middle is not always where you sit, though, and a wall-mounted camera is
rarely aimed where you would have chosen. Pan slides the crop window around
inside the frame, so the zoomed view can sit off to one side without anyone
re-aiming the hardware.

Pan is a fraction of the *slack* — the part of the frame the window does not
cover — rather than a pixel offset: 0 centres, -1 is flush against the left or
top edge, +1 flush against the right or bottom. Three things fall out of that
choice, all of which a pixel offset would cost us: the window can never leave
the frame, the pan stays meaningful when the zoom changes (so framing survives
a Crop-in tweak instead of flying off), and at zoom 1 there is no slack, so pan
is simply inert rather than out of range.

The SAME zoom AND pan must be applied to the rgb frame and its aligned depth
map: the engine's depth sampler maps tracked pixel coordinates into the depth
array by relative scale, which survives any equal-fraction crop — but only as
long as both windows also sit at the same relative offset.

Pure numpy; returns views (no copy).
"""
from __future__ import annotations

import numpy as np

_MIN_SIDE = 32  # never crop below this many pixels per side


def _origin(full: int, side: int, pan: float) -> int:
    """Left (or top) edge of a `side`-long window on a `full`-long axis.

    `pan` runs -1..+1 across the slack. pan=0 reproduces a centred crop
    exactly — floor of half the slack — so existing setups do not shift by a
    pixel when this grows a pan argument."""
    slack = full - side
    if slack <= 0:
        return 0
    pan = min(max(pan, -1.0), 1.0)
    return min(max(int(slack * (1.0 + pan) / 2.0), 0), slack)


def crop_view(arr: np.ndarray, zoom: float, pan_x: float = 0.0,
              pan_y: float = 0.0) -> np.ndarray:
    """The 1/zoom window of an HxW or HxWxC array, positioned by pan.

    zoom <= 1 is the identity: a full-frame window has no slack to pan within.
    Returns a view into the input."""
    if zoom <= 1.0:
        return arr
    h, w = arr.shape[:2]
    cw = max(int(round(w / zoom)), min(_MIN_SIDE, w))
    ch = max(int(round(h / zoom)), min(_MIN_SIDE, h))
    x0 = _origin(w, cw, pan_x)
    y0 = _origin(h, ch, pan_y)
    return arr[y0:y0 + ch, x0:x0 + cw]
