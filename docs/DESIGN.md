# Design notes

## Goals

1. **Latency**: hand motion → soundbar volume change fast enough to feel
   connected (~150–250 ms wall-to-wall, with visual knob position updating at
   camera rate).
2. **No false positives**: a media control that misfires when you talk with
   your hands is worse than no control at all.
3. **Feel like hardware**: absolute positioning, ratcheting, deadband, and
   smoothing tuned like a real knob's detent + damping.
4. **Run forever**: survive Kinect USB stalls, HA restarts, network blips
   without human intervention.

## Pipeline & threading

```
capture thread ──▶ LatestFrameSlot ──▶ vision loop (main thread) ──▶ asyncio thread
 (device I/O)      (1-slot, newest      MediaPipe + gesture FSM       controller +
                    frame wins)                                       HA WebSocket + web UI
```

- **LatestFrameSlot** is the latency keystone: a single-slot handoff where a new
  frame *replaces* an unprocessed old one. The tracker always works on the
  freshest image; backpressure never builds a queue of stale frames.
- MediaPipe Tasks `HandLandmarker` runs in **VIDEO mode on CPU**. Measured
  ~8–15 ms/frame at 640 px on a desktop Zen 2 core. The Linux GPU delegate in
  current mediapipe wheels is broken (upstream #6231) and unnecessary at this
  resolution. Timestamps are forced strictly increasing (graph wedges otherwise).
- The **asyncio thread** owns everything network: HA WebSocket client,
  controller queue, uvicorn. Vision → controller handoff is
  `loop.call_soon_threadsafe` into an `asyncio.Queue` — no locks shared with the
  hot path.

## Rotation estimation (the knob)

Naive approach — track the absolute angle of one hand vector — fails: different
landmarks disagree about "the hand's angle" and individual landmarks glitch.

Instead, per frame while gripped:

1. Take four rigid hand vectors: wrist→middleMCP, wrist→indexMCP,
   wrist→pinkyMCP, indexMCP→pinkyMCP.
2. Compute each vector's **frame-to-frame angular delta** (wrapped to ±180°).
   Deltas don't require the vectors to agree on an absolute angle — only on how
   much the hand rotated this frame, which for a rigid palm they do.
3. Take the **median** of the four deltas → any single glitching landmark is
   outvoted. Reject frames with |median| > 40° (physically implausible at 30 fps).
4. Accumulate, then smooth with a **One Euro filter** (min_cutoff 1.2 Hz,
   β 0.015): heavy smoothing at rest (no jitter), light during fast twists
   (no lag). This is the same filter class used by VR hand-tracking UIs.
5. Subtract a 3° **deadband** anchored at grip time so grabbing the knob never
   moves the volume.

Sign convention: image coordinates have y down, so `atan2(dy, dx)` deltas are
positive-clockwise *on screen*. Frames are mirrored (selfie view), which makes
screen-clockwise equal user-clockwise → positive = volume up. `knob.invert`
exists for the unmirrored case.

### Engagement FSM

```
IDLE ──pinch<0.42 & palm slow, 3 frames──▶ ENGAGED ──pinch>0.65, 5 frames──▶ IDLE
                                            │  hand lost >0.3s ────────────▶ IDLE
```

- `pinch = |thumbTip−indexTip| / |wrist−middleMCP|` — scale-invariant, so it
  works at any distance.
- Hysteresis (0.42 engage / 0.65 release) + debounce frames kill flicker.
- `max_engage_speed` blocks gripping while the hand is moving fast — a pinch
  during a swipe or a reach never grabs the knob.
- Hand dropouts shorter than 0.3 s keep the grip (MediaPipe loses a frame now
  and then); the rotation reference is re-based on reacquisition so the knob
  never jumps.
- Regripping ratchets: rotation only accumulates while pinched, exactly like
  a physical knob.

## Volume mapping

On engage the controller anchors at the entity's **current** volume (from the
live state cache — includes changes made from the Bose app/remote). Then:

```
target = clamp(anchor + rotation_deg / full_scale_deg, 0, max_volume)
```

Absolute targets are the key robustness choice: a lost or rate-limited command
costs nothing, because the next send lands exactly where the hand is. Sends are
coalesced to ≤10/s and quantised to 0.01 (the Bose Music API is integer 0–100;
finer sends are no-ops). Release flushes the final value immediately.

If HA doesn't know the volume (entity just restarted), the controller degrades
to relative `volume_up`/`volume_down` detents rather than doing nothing.

## Swipe detection

Sliding 0.35 s window over the palm centre track. Fires only when **all** hold:

- open palm (≥4 fingers extended) in essentially every window sample;
- horizontal travel ≥18% of frame width, mean speed ≥0.8 widths/s;
- |vertical| < 0.6 × |horizontal|;
- the hand has been visible ≥0.35 s (kills hand-entering-frame false skips);
- knob not engaged, and 0.4 s after any knob release (release flick ≠ swipe);
- 0.8 s cooldown.

## Depth gating (Kinect's actual superpower)

A webcam can't tell your hand from someone on the TV or a person behind the
couch. The Kinect's depth stream — registered to the RGB image so `depth[y,x]`
matches `rgb[y,x]` — lets the engine take the median depth of five palm
landmarks and drop any hand outside `[0.5 m, 3.0 m]` *before* gesture logic
runs. On v1 this comes from `DEPTH_REGISTERED` (mm, hardware-aligned); on v2
from `Registration.apply(..., with_big_depth=True)` (1920×1082 → cropped,
nearest-neighbour-scaled alongside the color frame).

Additionally `min_hand_frac` rejects hands too small to be near, which also
works in webcam mode.

## Home Assistant link

- One persistent WebSocket, authenticated once — service calls cost one send
  on an open TCP connection (~5–30 ms LAN round trip), not an HTTP handshake.
- Entity state kept fresh via `subscribe_trigger` (server-side filtered per
  entity; needs admin token) with automatic fallback to `state_changed` events
  filtered client-side.
- Exponential backoff reconnect (1 → 15 s cap); message IDs reset per
  connection; pending calls fail soft (logged, never crash the controller).
- Volume → Bose entity (local control, works for every source). Transport →
  Spotify entity (controls Spotify Connect wherever it plays). Both
  configurable independently.

## Failure handling

| Failure | Behaviour |
|---|---|
| Kinect v2 USB stall (no frames 5 s) | `CaptureError` → process exits code 3 → container restart policy revives it (libfreenect2's stall is only recoverable by reopening the device) |
| Vision starvation (15 s) | same exit-and-restart path |
| HA down / restarting | reconnect loop with backoff; gestures during the gap are logged and dropped harmlessly (absolute targets — nothing drifts) |
| Entity unavailable | anchor missing → relative detent fallback |
| MediaPipe timestamp violation | monotonic-forcing counter prevents it by construction |
| Model file missing | auto-downloaded at startup (also baked into the image) |

## What was deliberately not done

- **Skeleton/body tracking** (old Kinect SDK style): unnecessary — hand
  landmarks from RGB are more precise for finger-level gestures, and the only
  thing depth is needed for is distance gating.
- **MediaPipe GestureRecognizer task**: one more model + canned gestures that
  don't include "knob twist"; the landmark-level FSM is simpler and testable.
- **CUDA depth pipeline for v2**: needs CUDA-samples headers CUDA 12 no longer
  ships, saves ~0.2 ms over OpenCL. Not worth the build fragility.
- **LIVE_STREAM mode / result callbacks**: VIDEO mode on our own thread with
  newest-frame-wins gives lower, more predictable latency.
