# kinect-knob — agent notes (Unraid deployment copy)

This checkout at `/mnt/user/appdata/kinect-knob` on the Unraid server IS the
live deployment: the running `kinect-knob` container is built from it via
docker compose. `deploy/UNRAID_AGENT.md` is the deployment runbook and wins
over the README where they differ.

## The change → deploy loop

1. Make the change; keep `pytest` green (`docker run --rm --entrypoint sh
   -v $PWD:/w -w /w kinect-knob:local -c "pip install -q pytest; python3 -m
   pytest -q"` runs the suite in the app image — the image's entrypoint is
   the app CLI, so it must be overridden, and pytest isn't preinstalled).
2. Commit and push (origin is the SSH form of the GitHub repo; the server's
   key must be registered on the account).
3. Redeploy: `docker compose up -d --build` — layer cache makes app-only
   changes fast (~1 min). First-ever build is ~10 min (compiles libfreenect2).
4. Verify per runbook §4: `docker logs kinect-knob` (expect kinect2 +
   OpenCL pipeline lines), `curl -s localhost:8420/api/state` (fps ≈ 30,
   `backend: kinect2`, `has_depth: true`).

## Whiteboard-sync integration

- `GET /api/snapshot` serves whiteboard-sync (port 8430). `frames=N` (2-32)
  stacks N consecutive color frames into a denoised "proper photo" via
  `capture_photo()` on the kinect2 backend (the freenect2 binding exposes NO
  exposure/gain control, so temporal stacking is the only real quality lever);
  `format=png` returns lossless PNG. Without params it's the old cached
  ~1s-stale JPEG. `X-Snapshot-Mode` header says which path served the request.
- Play/pause is a held OPEN PALM FACING the camera (`playpause.*` config,
  `palm_facing_score` in the engine); `KK_PLAYPAUSE_POSE=fist` restores the
  old fist trigger. Old `fist.*` keys in `data/tuning.json` are ignored after
  this rename (deliberate — they were tuned for the fist pose).
- The facing SIGN was field-verified 2026-07-07 (the derivation-from-docs sign
  was inverted: the back of the hand triggered). Don't re-derive it from
  MediaPipe docs — if it reads inverted, flip `playpause.invert_facing`.
  Held objects are rejected via `finger_spread` + a palm-vs-wrist depth gap;
  live values show in `/api/state` engine.extra (facing / spread / obj_gap).
- Gate-level busy-hand rejection (`gate.object_gap_m`, depth-only): a hand
  holding an object can't engage the knob / swipe / play-pause, and a free
  hand steals primary from a holding one. Shape checks can't gate the knob
  (the pinch IS a holding shape) — only depth object evidence does. Thin
  objects pinch-held (toothbrush tip-grip) may still slip through if the
  handle misses the palm probes; the gap threshold + `gate.busy_linger_s`
  are dashboard tunables, and `holding`/`obj_gap` live in engine.extra —
  field-tune there rather than re-deriving geometry.

## Motion blur / exposure (2026-07-09)

- Swipes were dying in dim light: color auto-exposure stretches the shutter
  (~33 ms, 15 fps) and fast hands smear. Fix is three-part: (1) shutter cap
  via `KK_EXPOSURE=semi:8` — the python binding never wrapped libfreenect2's
  exposure API, so `native/kk_exposure.cpp` (built into the image as
  libkk_exposure.so) calls it via the binding's raw `Freenect2Device*`
  (`Device._c_object`); (2) `capture.low_light_boost` auto-gamma re-brightens
  the now-darker frames before MediaPipe; (3) `gate.lost_grace_s` keeps hand
  identity/presence/swipe history through 1-2 frame blur dropouts (previously
  ONE empty frame mid-swipe wiped history AND reset the 0.35 s presence gate,
  making fast swipes physically impossible).
- Field-verify exposure took effect via `docker logs`: "color sensor exposure
  X ms, gain Y" — dim room should read ≈8 ms with semi:8, not ~30. Whiteboard
  snapshot quality in dim light degrades with a capped shutter (darker/noisier
  stacks); if that bites, revert KK_EXPOSURE=auto rather than disabling boost.

## Presence gating + camera supervision (2026-09-17)

- **Presence** (`presence.py`, `/api/presence`, `KK_PRESENCE_*`, dashboard
  group "Presence"): depth background subtraction says whether a human is
  actually in front of the camera. The vision loop skips MediaPipe entirely
  while the room is empty (`shared.idle`, `proc_ms` 0, `/healthz` reports
  `idle`) and the kinect2 backend drops depth registration from every 2nd
  frame to every 15th. The CAMERA NEVER STOPS STREAMING while idle —
  whiteboard-sync photographs the boards through `/api/snapshot` exactly
  when nobody is home, so idle must mean "not inferring", never "not
  capturing". Wake is one probe (≤0.5 s); release waits `linger_s` (20 s).
- The background model is FROZEN in the "something arrived" direction while
  presence is true, otherwise somebody sitting still would be absorbed into
  the wall. Furniture still has to be absorbed eventually or the pipeline
  never sleeps, so the discriminator is MOTION, not time: only a scene that
  has not moved at all for `static_absorb_s` (180 s; 300 s for regions) gets
  built into the background. Don't "simplify" this to a plain EMA — that
  regression is invisible for minutes and then silently un-gates everything.
- `GET /api/presence?x1&y1&x2&y2` answers the same question about a RECTANGLE
  in unmirrored full-res colour coordinates (what `/api/snapshot` serves),
  returning the fraction of it sitting in front of its learned plane. This
  is what whiteboard-sync gates scans on. It exists because `region_depth`'s
  p10-vs-p90 test needs ~10% coverage to move and a head in front of a board
  half is a few percent — that gap is how half-occluded lines kept becoming
  Todoist tasks. `POST /api/presence/relearn` after the camera or furniture
  moves.
- Region occupancy runs on a 4x-decimated copy of the aligned map, rate
  limited to 1 Hz. Full-res every registration would cost ~25 ms/frame.

## The boot loop, and why the process no longer exits (2026-09-17)

- Found at 2204 restarts: the Kinect had dropped off the USB bus, `auto`
  detection fell back to **webcam** (this host has none), `create_capture`
  raised, the process exited 3, and Docker restarted it 61 s later — forever,
  each cycle paying a full libfreenect2 + OpenCL + MediaPipe startup.
- Capture is now acquired and supervised INSIDE `App.run`, and the asyncio
  thread (web server, HA client, controller) outlives it. Three cases:
  **absent** (nothing on the bus — `DeviceAbsent`) waits in-process forever,
  polling every 5 s, because no restart can summon a USB device; **failed /
  stalled** re-opens in-process; **failing repeatedly** (> `_MAX_RECYCLES`=4
  inside `_RECYCLE_WINDOW_S`=10 min) still exits 3 so the restart policy
  power-cycles the USB stack. Keep that last escape hatch.
- `KK_AUTO_FALLBACK=true` restores the old webcam fallback. Leave it off here.
- `/healthz` returns 200 with `status: "waiting for camera"` while deliberately
  waiting — the process is fine and restarting it changes nothing. It still
  503s on a camera that should be delivering frames and isn't.
- The dashboard chip row shows camera state, presence and recycle count, so
  "idling on purpose" is never mistaken for "dead".

## Recycles that actually recycle, and the night off window (2026-10-26)

- **The in-process recycle never worked until this date.** `stop()` only
  stopped the stream: the binding's `Device` has no finaliser and nothing
  called `close()`, so libfreenect2 kept the device in its open list and the
  "reopen" was handed the same half-dead handle (`device @4:5 is already be
  open!` — no USB reset, no re-enumeration, a new OpenCL pipeline leaked per
  attempt). Recycles failed within seconds, burned the 4/10 min budget and
  exited 3: 227 container restarts, each one flashing the sensor.
  `KinectV2Capture.stop()` now stops, CLOSES, and disposes the binding's
  process-wide `Freenect2` context (`_dispose_freenect2_context`), so the next
  open is a genuinely fresh start. Teardown joins the capture thread FIRST.
- **Half-started devices (`DeviceWedged`).** libfreenect2's `start()` returns
  false when every depth/IR submission fails, but leaves its 20 colour
  transfers in flight (the binding discarded the return value; we call
  `lib.freenect2_device_start` directly now). Such a device can be neither
  stopped (no-op) nor closed (frees in-flight transfers) — the process exits
  3 and the kernel reaps the URBs. Those stuck transfers were the 20
  `LIBUSB_ERROR_BUSY` lines on every failed reopen.
- Failure exits are remembered in `data/capture_exits.json` (30 min): a
  restart after recent failure exits rests 30 s, doubling to 10 min, before
  opening the sensor. In-process reopens back off 3/10/30/60 s.
- Presence models (scene + whiteboard region planes) now SURVIVE recycles —
  relearning re-seeded the "empty board" from whoever sat in front of it.
  `POST /api/presence/relearn` is the explicit reset.
- **Off window** `KK_OFF_WINDOW=20:00-06:30`, `KK_TZ=America/New_York`
  (compose default; set `KK_OFF_WINDOW=` empty in .env to disable). Every
  stall storm in the 2026-10-14..26 logs fell 21:00-07:00 Eastern. During the
  window the device is closed (`camera: "off"`), `/healthz` is 200 `scheduled
  off`, and `/api/snapshot`, `/api/region_depth`, region `/api/presence`
  return **503 `{"camera": "off"}`** — whiteboard-sync skips on that. At the
  window end the process exits 0 and the container restarts fresh: a clean
  daily restart, so nothing compounds across days.
- `POST /api/camera/restart` (dashboard "Restart camera") closes and reopens
  the sensor properly without counting against the failure budget.
- Snapshots never serve a frame older than 5 s; any non-`ok` camera is 503.
- The underlying USB stall (depth/IR iso transfers retiring one
  `LIBUSB_ERROR_IO` at a time until the pool of 60 runs dry) is unchanged —
  that is the Matisse CPU xHCI (09:00.3); a Renesas/Intel PCIe USB3 card
  remains the hardware fix.

## Sharp edges

- **Build fixes live in Dockerfile on purpose** (setuptools upgrade — Ubuntu
  22.04's setuptools predates PEP 621 and silently builds an empty package
  without it; libegl1/libgles2 for the GPU pipeline). Upstreamed in 37d060a;
  don't "simplify" them away.
- `.env` is gitignored and holds the box's real config (HA entities, GPU UUID
  pin). Never commit it; never log the HA token.
- Don't pin a `/dev/...` device path in compose and don't drop the restart
  policy — see runbook §3.3/§9 (Kinect re-enumerates; restart is the designed
  USB-stall recovery).
- The Kinect stalls (`LIBUSB_ERROR_IO`) on this host's AMD USB controller
  (daytime ~1-4/h, overnight storms) — the in-process recycle recovers it
  (properly since 2026-10-26, see above). If gesture
  sessions feel choppy, that's why; hardware fix is an Intel/Renesas PCIe USB3
  card (runbook §2/§6).
