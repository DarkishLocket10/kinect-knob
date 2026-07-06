# Unraid deployment runbook — for the agent deploying this stack

You are deploying **kinect-knob**: a Dockerized service that watches an Xbox One
Kinect (v2) and turns hand gestures into Home Assistant calls (Bose Soundbar 700
volume, Spotify track skip). This document is your complete brief; it assumes you
have shell access to the Unraid host and can reach Home Assistant. Follow it in
order. The general README covers the product; **this file is the deployment
truth** and it wins where they differ.

## 1. Current state — what is already done and verified (2026-07-06)

| Item | Status |
|---|---|
| Unit tests | 65/65 pass |
| Full pipeline in dry-run (webcam backend, web UI, `/api/action` → controller → volume math) | verified on Yash's Windows PC |
| Kinect hardware | **verified working**: Kinect v2 (Xbox One), USB `045e:02c4` + adapter hub `045e:02d9`. Microsoft Configuration Verifier confirmed *Kinect Connected* and *depth stream at target frame rate*. (Its "USB Controller" red X is a cosmetic whitelist warning — streams passed.) |
| This Docker stack on Unraid | **never run — that is your job** |
| IR night mode (`KK_IR_MODE`, auto-switch to active-IR tracking in the dark) | new feature, unit-tested, **not yet tested on real hardware** — verify it in step 7 |

The Kinect was verified on Yash's Windows desktop and must now be **physically
moved to the Unraid server**. If `lsusb` shows no `045e:02c4`, the sensor is not
plugged in / not powered — stop and tell Yash; nothing you can do in software
fixes that.

## 2. Hard requirements (check before building)

1. **USB port**: the Kinect must be in an **Intel or Renesas USB 3.0 port** on
   the server. ASMedia controllers (most add-on cards) do not work with the v2.
   The adapter's 12 V power brick must be connected.
2. **NVIDIA runtime**: `docker info 2>/dev/null | grep -i nvidia` must show the
   nvidia runtime (Unraid **Nvidia-Driver** plugin). The host has two GTX 1080
   Tis; the driver must be on the **580.xx branch or older** (last branch with
   Pascal support). Check: `nvidia-smi` on the host.
3. **Home Assistant prerequisites** (in HA, not on Unraid):
   - Bose Soundbar 700 via the **cavefire/Bose-Homeassistant** HACS integration
     (the official SoundTouch integration does NOT work with this soundbar).
     Expected entity: `media_player.bose_soundbar_700` — verify the exact ID.
   - **Spotify** integration for track skip. Entity looks like
     `media_player.spotify_<account>` — verify the exact ID.
   - A **long-lived access token** (HA → profile → Security). Get this from
     Yash if you don't already hold HA credentials. Never commit it.

## 3. Install

```bash
mkdir -p /mnt/user/appdata/kinect-knob
git clone https://github.com/DarkishLocket10/kinect-knob.git /mnt/user/appdata/kinect-knob
cd /mnt/user/appdata/kinect-knob
```

### 3.1 Create `.env` (gitignored — it does NOT come with the clone)

`cp .env.example .env`, then set these. Known-correct values are filled in;
three values you must obtain:

```bash
KK_HA_URL=http://<HA-IP>:8123                     # OBTAIN: Yash's HA address
KK_HA_TOKEN=<long-lived-token>                    # OBTAIN: from Yash / HA profile
KK_VOLUME_ENTITY=media_player.bose_soundbar_700   # verify exact ID in HA
KK_MEDIA_ENTITY=media_player.spotify_<account>    # OBTAIN: exact ID from HA
KK_MAX_VOLUME=0.9        # safety ceiling — do not raise without Yash's say-so
KK_BACKEND=auto          # auto-detects the v2 on the USB bus
KK_IR_MODE=auto          # night mode: IR tracking when the room is dark
GPU_UUID=<uuid>          # `nvidia-smi -L` → pin ONE 1080 Ti by UUID (not index)
```

Verify entities exist before first start (typos are the #1 wiring failure):

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://<HA-IP>:8123/api/states \
  | grep -o '"entity_id":"media_player[^"]*"' | sort -u
```

### 3.2 Host udev rules (container can't open the Kinect without them)

```bash
cp deploy/99-kinect.rules /etc/udev/rules.d/
udevadm control --reload-rules && udevadm trigger
# Unraid's rootfs is RAM — persist across reboots:
echo 'cp /mnt/user/appdata/kinect-knob/deploy/99-kinect.rules /etc/udev/rules.d/ && udevadm control --reload-rules && udevadm trigger' >> /boot/config/go
```

### 3.3 Build + start

Preferred: **Compose Manager Plus** plugin → new stack → point at this folder →
*Compose Up*. Equivalent CLI: `docker compose up -d --build`.
First build ≈ 10 min (compiles libfreenect2 with the OpenCL depth processor).

Do not "optimize" the compose file: the whole-bus mount `/dev/bus/usb` (not a
single device node) and `device_cgroup_rules: c 189:* rmw` are required because
the Kinect **re-enumerates** on driver init and after USB stalls; a pinned
device path breaks on the first stall. `restart: unless-stopped` is the
intended recovery mechanism for the v2's known long-run USB stalls — the
service deliberately exits non-zero when frames stop for 5 s.

## 4. Verify — in this order, before any gesture testing

```bash
docker logs kinect-knob 2>&1 | head -50
```

Expect, in the startup log:
1. `auto-detected kinect2 on the USB bus`
2. `Kinect v2 streaming (1080p color + ToF depth, GPU depth pipeline)`
3. A libfreenect2 line naming the **OpenCL** depth packet processor. If it says
   CPU pipeline, depth decode silently degraded — see §6.

Then:

```bash
docker exec kinect-knob clinfo -l          # must list ≥1 OpenCL platform
curl -s http://localhost:8420/healthz       # {"status":"ok"}
curl -s http://localhost:8420/api/state
```

In `/api/state` expect: `"backend":"kinect2"`, `"has_depth":true`,
`"fps"` ≈ 30, `"proc_ms"` < 25, and `"controller"` showing
`"mode":"live"`-style HA fields with `"ha_connected":true`.

## 5. Verify the HA wiring with the built-in test actions

These fire the **real** HA services — confirm the soundbar/Spotify respond
before trusting gestures. Warn Yash audio may change, keep volume low:

```bash
curl -s -X POST http://localhost:8420/api/action -H 'Content-Type: application/json' -d '{"action":"volume_up"}'
curl -s -X POST http://localhost:8420/api/action -H 'Content-Type: application/json' -d '{"action":"next"}'
```

Success = soundbar volume moves / track skips, and the events appear in
`/api/state → controller.events`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Compose fails: `unknown runtime: nvidia` | Nvidia-Driver plugin not installed / Docker not restarted after install |
| `clinfo` shows 0 platforms | `runtime: nvidia` missing or `NVIDIA_VISIBLE_DEVICES` unset → depth falls back to CPU (slow). Check compose env + plugin |
| `Kinect v2 not found` in logs | Not powered, wrong USB port (ASMedia), or udev rules not applied (§3.2) |
| `entities not found in HA` | Entity ID typo — copy exact IDs from HA Developer Tools → States |
| Container restarts every few seconds at first boot | Normal once or twice (device re-enumeration). Persistent crash-loop → read `docker logs` |
| Frames stop after hours/days, container restarts itself | Known libfreenect2 USB stall; restart policy is the designed recovery. If frequent: move to an Intel-chipset port, suspect a weak clone adapter power brick |
| Low fps with v2 | Confirm OpenCL pipeline in startup log (not CPU) |

## 7. Final acceptance (with Yash, at the couch)

1. Web UI at `http://SERVER-IP:8420` shows chips: **camera: kinect2**,
   **depth: on**, **home assistant: connected**, ~30 fps.
2. Pinch (thumb+index) and twist → volume dial follows; release → stops.
3. Open-palm swipe left/right → previous/next track.
4. Hand beyond 3 m → ignored (`gated_out` shows the reason in the UI).
5. **IR night mode** (new, first hardware test): lights off → within ~1 s the
   UI shows the **IR night mode** chip, `/api/state → ir_active:true`, and
   gestures still track. Lights on → chip clears. If tracking is poor in IR,
   note it for Yash — thresholds live in `src/kinectknob/capture/ir.py`
   (`DARK_LUMA`/`BRIGHT_LUMA`/`DWELL_FRAMES`) and gesture tuning in
   `config.example.yaml`.
6. `/healthz` returns ok (Docker healthcheck uses it; HA can alert on it too).

## 8. Guardrails

- Never commit `.env` or the token; never echo the token into logs.
- Keep `KK_MAX_VOLUME=0.9` unless Yash explicitly raises it.
- Don't switch the compose file to a pinned `/dev/...` device path (see §3.3).
- Don't disable the restart policy — it's the stall recovery.
- GPU: pin by **UUID**, not index (indices reshuffle across reboots).
