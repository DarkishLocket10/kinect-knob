"""Debug overlay rendering for the MJPEG stream (and --preview window)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import EngineSnapshot, Hand, INDEX_TIP, THUMB_TIP

_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_STATE_COLORS = {
    "idle": (160, 160, 160),
    "engaging": (0, 200, 255),
    "engaged": (0, 230, 120),
}


def render(
    rgb: np.ndarray,
    hands: list[Hand],
    snap: EngineSnapshot,
    volume: Optional[float],
) -> np.ndarray:
    """Returns a BGR frame with the overlay drawn (input is not modified)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    color = _STATE_COLORS.get(snap.state, (160, 160, 160))

    for hand in hands:
        pts = hand.pts.astype(int)
        for a, b in _CONNECTIONS:
            cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), (90, 90, 90), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(bgr, tuple(p), 2, (200, 200, 200), -1, cv2.LINE_AA)
        # Highlight the pinch pair.
        cv2.line(bgr, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), color, 2, cv2.LINE_AA)

    # Knob dial: shows accumulated rotation while engaged.
    if snap.hand_present:
        cx, cy = int(snap.palm_xy[0]), int(snap.palm_xy[1])
        cv2.circle(bgr, (cx, cy), 36, color, 2, cv2.LINE_AA)
        if snap.state == "engaged":
            ang = np.radians(snap.angle_deg - 90.0)
            tip = (int(cx + 34 * np.cos(ang)), int(cy + 34 * np.sin(ang)))
            cv2.line(bgr, (cx, cy), tip, color, 3, cv2.LINE_AA)

    # HUD strip.
    hud = [
        f"{snap.state.upper()}",
        f"pinch {snap.pinch_ratio:.2f}",
        f"pose {snap.openness or '-'}",
    ]
    if snap.state == "engaged":
        hud.append(f"angle {snap.angle_deg:+.0f} deg")
    if snap.hand_depth_m is not None:
        hud.append(f"depth {snap.hand_depth_m:.2f} m")
    if volume is not None:
        hud.append(f"vol {volume:.0%}")
    cv2.rectangle(bgr, (0, 0), (w, 24), (25, 25, 25), -1)
    cv2.putText(
        bgr, "   ".join(hud), (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )

    # Volume bar along the bottom.
    if volume is not None:
        vol = max(0.0, min(1.0, volume))
        cv2.rectangle(bgr, (0, h - 6), (w, h), (40, 40, 40), -1)
        cv2.rectangle(bgr, (0, h - 6), (int(w * vol), h), (0, 230, 120), -1)

    if snap.last_event:
        cv2.putText(
            bgr, snap.last_event, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 200, 255), 1, cv2.LINE_AA,
        )
    return bgr
