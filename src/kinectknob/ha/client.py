"""Home Assistant WebSocket client.

One persistent WebSocket connection (auth once, then fire service calls with
no per-command HTTP/TLS setup cost — this is what keeps gesture->soundbar
latency in the tens of milliseconds).

Structure: after the auth handshake a dedicated **reader task** owns the
socket's receive side and resolves the futures of in-flight requests; priming
(get_states) and subscriptions then run as ordinary requests against it. The
client only becomes visible to callers (``connected``/``_ws``) once fully
subscribed — a gesture arriving mid-handshake is dropped, never written into
the auth phase.

Entity states are tracked via ``subscribe_trigger`` (server-side filtered;
requires an admin token, which a personal long-lived token normally is) with
automatic fallback to a client-filtered ``state_changed`` subscription — so
the locally cached volume always matches reality, even when it's changed from
the Bose app / remote / HA UI.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import websockets

log = logging.getLogger("kk.ha")


class HAClient:
    def __init__(self, url: str, token: str, entities: list[str]):
        # Accept http(s):// or ws(s):// and normalise to the websocket endpoint.
        base = url.rstrip("/")
        if base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        elif base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        self._ws_url = base + "/api/websocket"
        self._token = token
        self._entities = [e for e in dict.fromkeys(entities) if e]
        self._ws: Optional[Any] = None  # published only after subscribe completes
        self._msg_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._trigger_subs: dict[int, str] = {}   # subscription id -> entity_id
        self.states: dict[str, dict[str, Any]] = {}  # entity_id -> {"state","attributes"}
        self._state_updated: dict[str, float] = {}   # entity_id -> time.monotonic()
        self.connected = False
        self.last_error = ""
        self._stop = False

    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Connect/auth/subscribe, dispatch messages, reconnect forever."""
        backoff = 1.0
        while not self._stop:
            read_task: Optional[asyncio.Task] = None
            try:
                async with websockets.connect(
                    self._ws_url, max_size=16 * 1024 * 1024, open_timeout=10, close_timeout=3
                ) as ws:
                    self._msg_id = 0          # ids are per-connection
                    self._trigger_subs = {}
                    await self._handshake(ws)
                    # Reader must run concurrently: request futures are resolved
                    # by it, including those sent by _subscribe_and_prime below.
                    read_task = asyncio.create_task(self._read_loop(ws))
                    await self._subscribe_and_prime(ws)
                    self._ws = ws             # now safe for call_service
                    self.connected = True
                    self.last_error = ""
                    backoff = 1.0
                    log.info("connected to Home Assistant at %s", self._ws_url)
                    await read_task           # ends when the connection drops
                    read_task = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect on anything
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("HA connection lost (%s); retrying in %.0fs", self.last_error, backoff)
            finally:
                self.connected = False
                self._ws = None
                if read_task is not None:
                    read_task.cancel()
                    try:
                        await read_task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("HA connection lost"))
                self._pending.clear()
            if self._stop:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def stop(self) -> None:
        self._stop = True

    async def _handshake(self, ws) -> None:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if hello.get("type") != "auth_required":
            raise ConnectionError(f"unexpected first message: {hello.get('type')}")
        # Auth-phase messages carry no id field.
        await ws.send(json.dumps({"type": "auth", "access_token": self._token}))
        reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if reply.get("type") != "auth_ok":
            raise ConnectionError(f"HA auth failed: {reply.get('message', reply.get('type'))}")

    async def _subscribe_and_prime(self, ws) -> None:
        # Prime the state cache.
        states = await self._request(ws, {"type": "get_states"})
        for st in states or []:
            if st.get("entity_id") in self._entities:
                self._store_state(st["entity_id"], st)
        missing = [e for e in self._entities if e not in self.states]
        if missing:
            log.warning("entities not found in HA: %s", ", ".join(missing))

        # Server-side filtered subscription per entity; fall back to the firehose.
        try:
            for ent in self._entities:
                msg_id, fut = await self._send(
                    ws,
                    {"type": "subscribe_trigger", "trigger": {"platform": "state", "entity_id": ent}},
                )
                # Register before awaiting the ack so an event racing the result
                # message can't be dropped.
                self._trigger_subs[msg_id] = ent
                try:
                    await asyncio.wait_for(fut, 10)
                except Exception:
                    self._trigger_subs.pop(msg_id, None)
                    raise
                finally:
                    self._pending.pop(msg_id, None)
        except Exception as exc:  # noqa: BLE001 — e.g. non-admin token
            log.info("subscribe_trigger unavailable (%s); using state_changed events", exc)
            self._trigger_subs = {}
            await self._request(ws, {"type": "subscribe_events", "event_type": "state_changed"})

    def _store_state(self, entity_id: str, state_obj: Optional[dict]) -> None:
        if state_obj:
            self.states[entity_id] = {
                "state": state_obj.get("state"),
                "attributes": state_obj.get("attributes", {}),
            }
            self._state_updated[entity_id] = time.monotonic()

    async def _read_loop(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "result":
                fut = self._pending.pop(msg.get("id"), None)
                if fut is not None and not fut.done():
                    if msg.get("success"):
                        fut.set_result(msg.get("result"))
                    else:
                        fut.set_exception(
                            RuntimeError(str(msg.get("error", {}).get("message", "call failed")))
                        )
            elif mtype == "event":
                sub_id = msg.get("id")
                event = msg.get("event", {})
                if sub_id in self._trigger_subs:
                    trigger = event.get("variables", {}).get("trigger", {})
                    self._store_state(self._trigger_subs[sub_id], trigger.get("to_state"))
                else:
                    data = event.get("data", {})
                    if data.get("entity_id") in self._entities:
                        self._store_state(data["entity_id"], data.get("new_state"))

    async def _send(self, ws, payload: dict) -> tuple[int, asyncio.Future]:
        """Allocate an id, register its future, send. Caller awaits the future."""
        self._msg_id += 1
        msg_id = self._msg_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        await ws.send(json.dumps({"id": msg_id, **payload}))
        return msg_id, fut

    async def _request(self, ws, payload: dict, timeout: float = 10.0) -> Any:
        msg_id, fut = await self._send(ws, payload)
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(msg_id, None)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def call_service(
        self, domain: str, service: str, entity_id: str, data: Optional[dict] = None
    ) -> bool:
        """Fire a service call. Returns False (and logs) instead of raising, so a
        dropped connection never crashes the controller."""
        ws = self._ws
        if ws is None:
            log.debug("dropping %s.%s — not connected", domain, service)
            return False
        payload: dict[str, Any] = {
            "type": "call_service",
            "domain": domain,
            "service": service,
            "target": {"entity_id": entity_id},
        }
        if data:
            payload["service_data"] = data
        try:
            t0 = time.monotonic()
            await self._request(ws, payload, timeout=5.0)
            log.debug("%s.%s(%s) ok in %.0f ms", domain, service, data, (time.monotonic() - t0) * 1e3)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("service call %s.%s failed: %s", domain, service, exc)
            return False

    def volume_level(self, entity_id: str) -> Optional[float]:
        st = self.states.get(entity_id)
        if not st:
            return None
        vol = st.get("attributes", {}).get("volume_level")
        try:
            return float(vol) if vol is not None else None
        except (TypeError, ValueError):
            return None

    def state_updated_at(self, entity_id: str) -> Optional[float]:
        """time.monotonic() of the last cache update for this entity."""
        return self._state_updated.get(entity_id)

    def entity_state(self, entity_id: str) -> Optional[str]:
        st = self.states.get(entity_id)
        return st.get("state") if st else None
