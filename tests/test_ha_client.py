"""HAClient integration tests against an in-process mock Home Assistant
WebSocket server speaking the real protocol (auth_required -> auth ->
auth_ok, get_states, subscribe_trigger, call_service, events).

Exists because of a review finding: the original client deadlocked on connect
(requests awaited futures that only the not-yet-started read loop resolved) —
invisible to every other test. This test fails loudly on any such regression.
"""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from kinectknob.ha.client import HAClient

ENTITY = "media_player.bose"


class MockHA:
    def __init__(self, trigger_subs_supported: bool = True):
        self.trigger_subs_supported = trigger_subs_supported
        self.calls: list[dict] = []
        self.trigger_sub_ids: list[int] = []
        self.event_sub_ids: list[int] = []
        self.conn = None

    async def handler(self, ws):
        self.conn = ws
        await ws.send(json.dumps({"type": "auth_required", "ha_version": "2026.7"}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth" or msg.get("access_token") != "good-token":
            await ws.send(json.dumps({"type": "auth_invalid", "message": "bad token"}))
            await ws.close()
            return
        await ws.send(json.dumps({"type": "auth_ok", "ha_version": "2026.7"}))
        async for raw in ws:
            m = json.loads(raw)
            mtype, mid = m.get("type"), m.get("id")
            if mtype == "get_states":
                await ws.send(json.dumps({
                    "id": mid, "type": "result", "success": True,
                    "result": [{
                        "entity_id": ENTITY, "state": "playing",
                        "attributes": {"volume_level": 0.42},
                    }],
                }))
            elif mtype == "subscribe_trigger":
                if self.trigger_subs_supported:
                    self.trigger_sub_ids.append(mid)
                    await ws.send(json.dumps(
                        {"id": mid, "type": "result", "success": True, "result": None}
                    ))
                else:
                    await ws.send(json.dumps({
                        "id": mid, "type": "result", "success": False,
                        "error": {"code": "unauthorized", "message": "admin required"},
                    }))
            elif mtype == "subscribe_events":
                self.event_sub_ids.append(mid)
                await ws.send(json.dumps(
                    {"id": mid, "type": "result", "success": True, "result": None}
                ))
            elif mtype == "call_service":
                self.calls.append(m)
                await ws.send(json.dumps({
                    "id": mid, "type": "result", "success": True,
                    "result": {"context": {"id": "abc"}},
                }))

    async def push_trigger_event(self, volume: float):
        await self.conn.send(json.dumps({
            "id": self.trigger_sub_ids[-1], "type": "event",
            "event": {"variables": {"trigger": {
                "to_state": {"entity_id": ENTITY, "state": "playing",
                             "attributes": {"volume_level": volume}},
            }}},
        }))

    async def push_state_changed(self, volume: float):
        await self.conn.send(json.dumps({
            "id": self.event_sub_ids[-1], "type": "event",
            "event": {"event_type": "state_changed", "data": {
                "entity_id": ENTITY,
                "new_state": {"entity_id": ENTITY, "state": "playing",
                              "attributes": {"volume_level": volume}},
            }},
        }))


async def _wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.02)


async def _run_with_client(mock: MockHA, body):
    async with websockets.serve(mock.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = HAClient(f"http://127.0.0.1:{port}", "good-token", [ENTITY])
        task = asyncio.create_task(client.run())
        try:
            await _wait_for(lambda: client.connected)
            await body(client)
        finally:
            client.stop()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


def test_connects_primes_and_calls_service():
    mock = MockHA()

    async def body(client: HAClient):
        # Priming worked: no deadlock, cache populated from get_states.
        assert client.volume_level(ENTITY) == 0.42
        assert client.entity_state(ENTITY) == "playing"
        # Round-trip service call.
        ok = await client.call_service(
            "media_player", "volume_set", ENTITY, {"volume_level": 0.55}
        )
        assert ok
        sent = mock.calls[-1]
        assert sent["domain"] == "media_player"
        assert sent["service"] == "volume_set"
        assert sent["target"] == {"entity_id": ENTITY}
        assert sent["service_data"] == {"volume_level": 0.55}
        # Live state update via the trigger subscription.
        await mock.push_trigger_event(0.77)
        await _wait_for(lambda: client.volume_level(ENTITY) == 0.77)
        assert client.state_updated_at(ENTITY) is not None

    asyncio.run(_run_with_client(mock, body))


def test_falls_back_to_state_changed_for_non_admin():
    mock = MockHA(trigger_subs_supported=False)

    async def body(client: HAClient):
        assert mock.event_sub_ids, "must fall back to subscribe_events"
        await mock.push_state_changed(0.31)
        await _wait_for(lambda: client.volume_level(ENTITY) == 0.31)

    asyncio.run(_run_with_client(mock, body))


def test_bad_token_never_connects():
    async def scenario():
        mock = MockHA()
        async with websockets.serve(mock.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = HAClient(f"http://127.0.0.1:{port}", "wrong-token", [ENTITY])
            task = asyncio.create_task(client.run())
            with pytest.raises(TimeoutError):
                await _wait_for(lambda: client.connected, timeout=1.0)
            assert client.last_error
            client.stop()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    asyncio.run(scenario())
