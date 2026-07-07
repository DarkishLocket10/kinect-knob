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
- The Kinect stalls (`LIBUSB_ERROR_IO`) every few minutes on this host's AMD
  USB controller as of 2026-07-06 — watchdog restart recovers it. If gesture
  sessions feel choppy, that's why; hardware fix is an Intel/Renesas PCIe USB3
  card (runbook §2/§6).
