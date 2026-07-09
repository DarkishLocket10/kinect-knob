# kinect-knob

Turn an **invisible volume knob** in the air. An Xbox Kinect on your Unraid server
watches your hand: **pinch your thumb and index finger** like you're gripping a
small knob, **twist** — the volume of your Bose Soundbar 700 follows your hand in
real time through Home Assistant. **Swipe an open palm** left or right to skip
Spotify tracks. Let go, wind back, re-grip and keep turning — it ratchets exactly
like a physical knob.

```
        Kinect (USB)                Unraid server (Docker)                LAN
  ┌──────────────────┐   ┌───────────────────────────────────────┐   ┌──────────────┐
  │ RGB 30fps        ├──▶│ capture thread (newest frame wins)    │   │ Home         │
  │ depth (mm,       │   │   ▼                                   │   │ Assistant    │
  │  RGB-aligned)    │   │ MediaPipe hand landmarks (CPU, ~10ms) │   │   ▼          │
  └──────────────────┘   │   ▼                                   │   │ Bose 700     │
                         │ gesture engine (knob / swipe FSM)     ├──▶│ (volume, ws) │
    GTX 1080 Ti ────────▶│   ▼                                   │   │ Spotify      │
    (OpenCL depth decode │ controller → HA WebSocket (persistent)│   │ (skip track) │
     for Kinect v2 only) └───────────────────────────────────────┘   └──────────────┘
```

**Latency budget** (gesture → soundbar): capture ≤33 ms + hand tracking ~8–15 ms
(CPU, Ryzen 3700X) + gesture engine <1 ms + coalesced send ≤100 ms + HA→Bose
(local WebSocket) ~30–100 ms ⇒ **typically 150–250 ms end-to-end**, with the knob
position itself updating every frame. Volume is sent as *absolute* targets, so no
command can ever be lost in a way that drifts the knob.

---

## 1. Hardware: which Kinect do you have?

| | **Kinect v1** (Xbox 360) | **Kinect v2** (Xbox One) |
|---|---|---|
| Looks like | small bar on a motorised tilt base | big fixed slab, wide dark glass front |
| Model number | 1414 / 1473 / 1517 | 1520 |
| Adapter needed | 12 V AC/USB splitter (~$10, plentiful) | **Kinect Adapter for Windows** (discontinued; used ~$60, clones ~$20 — beware weak clone power bricks, they cause random disconnects) |
| Connection | USB 2.0 | USB 3.0 (**Intel/Renesas controller; ASMedia does not work**) |
| Depth quality at 1–3 m | good (cm-level) | excellent (mm-level ToF), works in the dark |
| GPU needed | no | yes — OpenCL depth decode (the 1080 Ti, ~1 ms/frame) |
| Driver stability 24/7 | boring-stable | good, occasional USB stall — auto-recovered (see §7) |

Both are fully supported; the container auto-detects whichever is plugged in
(`KK_BACKEND=auto`). **The Kinect must be plugged into the Unraid server itself**
(the box running this container) and pointed at where you sit.

> Two 1080 Tis note: run `nvidia-smi -L` on Unraid and put one card's UUID in
> `GPU_UUID` — UUIDs are stable across reboots, indices are not. The GPU is only
> used for Kinect v2 depth decode; hand tracking is deliberately CPU (MediaPipe's
> Linux GPU delegate is broken in current wheels, and CPU is ~10 ms/frame anyway).
> Keep the Unraid NVIDIA driver on the **580.xx branch or older** — it's the last
> branch that supports Pascal cards.

## 2. Home Assistant setup (one-time)

1. **Bose Soundbar 700** — the official SoundTouch integration does *not* work
   (the 700 is a Bose Music family device, no port-8090 API). Install
   [cavefire/Bose-Homeassistant](https://github.com/cavefire/Bose-Homeassistant)
   via **HACS** (v1.2.3+; it controls the soundbar 100% locally over WebSocket
   after a one-time Bose account login). You get `media_player.bose_soundbar_700`
   → this is your **volume entity**.
2. **Spotify** — add the official [Spotify integration](https://www.home-assistant.io/integrations/spotify/)
   (needs Spotify Premium + a Spotify developer app). You get
   `media_player.spotify_<account>` → this is your **media entity** for
   next/previous/play-pause: it controls whatever Spotify Connect is playing on,
   including the soundbar, regardless of how playback was started.
   *(No Spotify? Set `KK_MEDIA_ENTITY` to the Bose entity — skip then works only
   for sources the soundbar itself controls, and never for TV/eARC audio.)*
3. **Token** — click your user name (bottom-left) → **Security** tab →
   **Long-lived access tokens** → Create. Valid 10 years.

## 3. Deploy on Unraid

> **Deploying with an AI agent?** Point it at
> [deploy/UNRAID_AGENT.md](deploy/UNRAID_AGENT.md) — a self-contained runbook
> with the verified hardware facts, exact install steps, `.env` recreation
> (it's gitignored and does not arrive with a clone), and the ordered
> verification checklist. It supersedes this section where they differ.

```bash
# on the Unraid box
mkdir -p /mnt/user/appdata/kinect-knob && cd /mnt/user/appdata/kinect-knob
# copy this project folder here (or git clone), then:
cp .env.example .env && nano .env          # HA URL, token, entities, GPU UUID
cp deploy/99-kinect.rules /etc/udev/rules.d/ && udevadm control --reload-rules && udevadm trigger
# Unraid's rootfs is RAM — persist the udev rules across reboots:
echo 'cp /mnt/user/appdata/kinect-knob/deploy/99-kinect.rules /etc/udev/rules.d/ && udevadm control --reload-rules && udevadm trigger' >> /boot/config/go
```

**Option A — Compose Manager Plus plugin (recommended):** install it from
Community Applications, add a stack pointing at this folder, hit *Compose Up*.
It builds the image from the Dockerfile on the box (first build ~10 min) and
restarts it on crash (`restart: unless-stopped` — this is also the automatic
recovery path for Kinect USB stalls).

**Option B — dockerMan template:** `docker build -t kinect-knob:local .` once in
the terminal, copy `deploy/unraid-template.xml` to
`/boot/config/plugins/dockerMan/templates-user/my-kinect-knob.xml`, then *Add
Container* → template *kinect-knob* and fill in the fields.

Open **http://SERVER-IP:8420** — live status, volume dial, tracking numbers, a
camera debug view with the skeleton/knob overlay, and test buttons that fire the
actual HA services so you can verify wiring before waving at anything.

## 4. Try it on your Windows PC first (no Kinect needed)

```powershell
.\scripts\dev-windows.ps1 -Preview     # uses your webcam, Python 3.10-3.12
```

Runs the identical pipeline with your webcam in **dry-run** mode (simulated
volume, nothing sent to HA) so you can learn/tune the gestures at your desk. Set
`KK_HA_URL`/`KK_HA_TOKEN`/`KK_VOLUME_ENTITY` env vars first to control the real
soundbar from your PC. Unit tests: `pip install pytest numpy pyyaml websockets`
then `python -m pytest tests` (any Python ≥3.10 — no Kinect/mediapipe needed).

## 5. The gestures

| Gesture | Action |
|---|---|
| **Pinch** (thumb+index together, like gripping a knob) | grab the volume knob |
| **Twist** while pinched — clockwise | volume up (270° = 0→100%, so ~27° per 10%) |
| **Twist** counter-clockwise | volume down |
| release pinch, wind back, pinch again | ratchet — keep turning past wrist range |
| **Open palm, swipe right** | next track |
| **Open palm, swipe left** | previous track |
| **Open palm facing the camera, hold 0.7 s** | play/pause |

Play/pause only counts when the palm actually FACES the lens (a handedness-aware
cross-product check) — the back of a raised hand, an edge-on palm, or a hand
waving past never toggles playback. A hand HOLDING something is rejected two
ways: bunched fingertips (a wrapped hand can't spread its fingers) and, via the
Kinect's depth, an object surface sitting closer to the camera than the wrist.
If palm/back read inverted on your camera, flip "Invert palm/back" in the
tuning UI. `KK_PLAYPAUSE_POSE=fist` restores the old closed-fist trigger;
`KK_PLAYPAUSE_ENABLED=false` turns it off.

Anti-misfire measures baked in: pinch must hold ~100 ms to engage (with
hysteresis to release); a 3° deadband means grabbing never nudges the volume;
single-frame tracking glitches are rejected by a 4-vector median; swipes require
an open palm, sustained speed, mostly-horizontal travel, and a hand that's been
visible ≥0.35 s (someone walking past can't skip your track); with a Kinect,
**hands outside 0.5–3 m are ignored entirely** via the depth camera; and
`KK_MAX_VOLUME` (default 0.9 in compose) hard-caps what a gesture can ever set.

**Busy-hand rejection (Kinect depth):** a hand holding an object — water
bottle, mug, toothbrush — is "busy" and can't grab the knob, swipe, or toggle
playback at all. Detection is object *evidence*, not hand shape (a grip on a
toothbrush is shaped exactly like the knob pinch): the held object's surface
sits well in front of the wrist plane over the palm area (`gate.object_gap_m`,
0 disables). While one hand is busy, your **other hand takes over
automatically** — raise it and it becomes the controlling hand even though
selection normally prefers the bigger/closer hand. Live `obj_gap` / `holding`
readouts appear in the web UI and `/api/state` for tuning.

**Works in the dark (Kinect v2):** the ToF sensor carries its own IR
illuminator, so when the room goes dark — movie night — the service
automatically switches hand tracking to the active-infrared camera and back
again when the lights come on (`KK_IR_MODE=auto`, the default; `always` /
`off` to force). The web UI shows an *IR night mode* chip while it's active.

**Fast hands stay sharp (Kinect v2):** in dim rooms the color camera's
auto-exposure stretches the shutter toward ~33 ms (and halves the stream to
15 fps) — a fast swipe smears into a blur the landmark model can't follow.
`KK_EXPOSURE=semi:8` caps the integration time at 8 ms while analog gain
floats, trading blur for a darker, noisier image; the built-in **low-light
boost** (auto-gamma, `KK_LOW_LIGHT_BOOST`, on by default) then lifts dim
frames back into the tracker's comfort zone. libfreenect2's exposure API was
never wrapped by the python binding — a small C++ bridge in the image
(`native/kk_exposure.cpp`) calls it with the binding's own device pointer.
The sensor's live shutter/gain shows in the logs (`color sensor exposure`).
On top of that, brief tracking dropouts no longer reset a gesture: a swipe
whose hand blurs out for a frame or two mid-motion still lands
(`gate.lost_grace_s`).

## 6. Tuning

Everything lives in [config.example.yaml](config.example.yaml) (env vars
override). The ones that matter:

- `knob.full_scale_deg` — 270 feels like a hi-fi amp; drop to 180 for a more
  sensitive knob.
- `knob.filter_min_cutoff` / `filter_beta` — One Euro filter; lower cutoff =
  smoother at rest, higher beta = snappier fast twists.
- `gate.depth_min_m` / `depth_max_m` — the "only me, from the couch" zone.
- `capture.crop` — crop-in zoom onto the frame centre (dashboard slider):
  everything outside the crop is invisible to tracking, and your hand fills
  more of the model's view. Drag it while watching Live tracking.
- `swipe.min_speed_frac` — raise it if skips feel too eager.
- `capture.mirror` — keep `true`; it makes clockwise-from-your-view = louder.
  If direction feels backwards, flip `knob.invert` instead.

Watch the live `pinch ratio`, `palm speed`, and `angle` numbers in the web UI
while adjusting — they're exactly what the engine sees.

## 7. Reliability & troubleshooting

- **Kinect v2 long-run USB stalls** (a known libfreenect2 issue): the service
  detects "no frames for 5 s", exits non-zero, and the container restart policy
  brings it back in seconds — no intervention. If stalls are frequent, move the
  Kinect to the Intel chipset USB3 ports (not an ASMedia add-on card) and
  suspect a clone adapter's power brick.
- **`clinfo` shows 0 platforms** in the container → check `--runtime=nvidia` is
  in Extra Parameters and `NVIDIA_VISIBLE_DEVICES` is set; otherwise depth decode
  silently falls back to CPU (log line says which pipeline was chosen).
- **`entities not found in HA`** in the logs → entity ID typo; copy the exact ID
  from HA Developer Tools → States.
- **Volume laggy** → check the web UI `fps` (should be ~30) and `ms/frame`
  (should be <25). If fps is low with Kinect v2, confirm the OpenCL pipeline is
  active (container log at startup).
- **Health**: `GET /healthz` (used by the Docker healthcheck), `GET /api/state`
  for full state — easy to alert on from HA itself.

## 8. Project layout

```
src/kinectknob/
  capture/        webcam / kinect_v1 (libfreenect) / kinect_v2 (freenect2) + auto-detect
  tracking/       MediaPipe HandLandmarker wrapper (VIDEO mode, CPU)
  gestures/       the knob/swipe/play-pause state machine (pure numpy, fully unit-tested)
  ha/             Home Assistant WebSocket client (persistent, auto-reconnect)
  controller.py   gesture events → coalesced volume_set / track skip
  web/            FastAPI status UI + MJPEG debug stream
  main.py         thread wiring, watchdog, model download
tests/            141 tests: engine geometry, swipe gates, busy-hand, low-light, crop, config, volume math
deploy/           Unraid template + host udev rules
docs/DESIGN.md    algorithm details & design rationale
```

MIT licensed. See [docs/DESIGN.md](docs/DESIGN.md) for how the rotation
estimator, One Euro filtering, and the engagement state machine work.
