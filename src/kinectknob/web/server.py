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
from .. import debugdraw

log = logging.getLogger("kk.web")

_STATIC = Path(__file__).parent / "static"


class ActionRequest(BaseModel):
    action: str


def create_app(cfg: AppConfig, shared: SharedState, controller: Controller) -> FastAPI:
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
