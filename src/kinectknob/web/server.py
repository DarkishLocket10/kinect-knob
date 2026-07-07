"""FastAPI status/debug server.

Runs inside the same asyncio loop as the HA client and controller, so the
action endpoints can await controller methods directly. The MJPEG debug
stream renders overlays only while at least one client is watching — zero
cost when nobody is looking.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import AppConfig
from ..controller import Controller
from ..state import SharedState
from ..tuning import Tuning
from .. import debugdraw

log = logging.getLogger("kk.web")

_STATIC = Path(__file__).parent / "static"


class ActionRequest(BaseModel):
    action: str


class TuningRequest(BaseModel):
    key: str
    value: bool | float


def create_app(
    cfg: AppConfig,
    shared: SharedState,
    controller: Controller,
    tuning: Tuning | None = None,
) -> FastAPI:
    app = FastAPI(title="kinect-knob", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index():
        return FileResponse(_STATIC / "index.html")

    @app.get("/healthz")
    async def healthz():
        if shared.healthy():
            return {"status": "ok"}
        return JSONResponse({"status": "no frames"}, status_code=503)

    @app.get("/api/state")
    async def state():
        data = shared.state_dict()
        data["controller"] = controller.snapshot()
        data["debug_stream"] = cfg.web.debug_stream
        return data

    @app.post("/api/action")
    async def action(req: ActionRequest):
        ok = await controller.manual(req.action)
        return {"ok": ok}

    @app.get("/api/tuning")
    async def tuning_get():
        if tuning is None:
            return JSONResponse({"error": "tuning unavailable"}, status_code=404)
        return {"tunables": tuning.schema()}

    @app.post("/api/tuning")
    async def tuning_set(req: TuningRequest):
        if tuning is None:
            return JSONResponse({"error": "tuning unavailable"}, status_code=404)
        try:
            applied = tuning.set_value(req.key, req.value)
        except KeyError:
            return JSONResponse({"error": f"unknown key {req.key!r}"}, status_code=400)
        return {"key": req.key, "value": applied}

    @app.post("/api/tuning/reset")
    async def tuning_reset():
        if tuning is None:
            return JSONResponse({"error": "tuning unavailable"}, status_code=404)
        tuning.reset()
        return {"ok": True}

    @app.get("/api/snapshot")
    async def snapshot(frames: int = 1, format: str = "jpeg", quality: int = 92):
        """Full-resolution UNMIRRORED photo — what the whiteboard reader
        consumes (the normal pipeline only keeps a downscaled copy, far too
        small for handwriting). ``frames=N`` (2-32) averages N consecutive
        frames into a denoised "proper photo" instead of grabbing the cached
        ~1s-old preview frame; ``format=png`` returns lossless PNG;
        ``quality`` is the JPEG quality otherwise."""
        frames = max(1, min(32, frames))
        quality = max(1, min(100, quality))
        fmt = "png" if format.strip().lower() == "png" else "jpeg"
        loop = asyncio.get_running_loop()
        bgr, mode = None, "cached"
        if frames > 1 and shared.photo_fn is not None:
            bgr = await loop.run_in_executor(
                None, shared.photo_fn, frames, frames / 30.0 + 3.0)
            if bgr is not None:
                mode = "stacked"
        if bgr is None:
            bgr, _ = shared.fullres()
        if bgr is None:
            return JSONResponse({"error": "no full-res frame yet"}, status_code=404)
        buf = await loop.run_in_executor(None, _encode_image, bgr, fmt, quality)
        if buf is None:
            return JSONResponse({"error": "encode failed"}, status_code=500)
        return Response(
            buf, media_type="image/png" if fmt == "png" else "image/jpeg",
            headers={"X-Snapshot-Mode": mode,
                     "X-Snapshot-Frames": str(frames if mode == "stacked" else 1)},
        )

    @app.get("/debug/stream")
    async def debug_stream():
        if not cfg.web.debug_stream:
            return Response("debug stream disabled", status_code=404)

        async def generate():
            boundary = b"--frame\r\n"
            loop = asyncio.get_running_loop()
            while True:
                rgb, hands, snap = shared.render_data()
                if rgb is not None:
                    volume = controller.snapshot().get("volume")
                    # Encode off the event loop; ~1-3 ms but keep the loop clean.
                    jpg = await loop.run_in_executor(
                        None, _render_jpeg, rgb, hands, snap, volume
                    )
                    if jpg is not None:
                        yield boundary
                        yield b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(1 / 12)

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def _render_jpeg(rgb, hands, snap, volume) -> bytes | None:
    bgr = debugdraw.render(rgb, hands, snap, volume)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else None


def _encode_image(bgr, fmt: str, quality: int) -> bytes | None:
    if fmt == "png":
        ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None
