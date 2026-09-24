# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The WebSocket protocol, driven exactly like the headset client drives it."""

import asyncio
import json
import struct

import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from superdex_quest_teleop.server import ServerConfig, TeleopServer
from superdex_quest_teleop.synthetic import ScriptedGrasp


def _decode(frame: bytes):
    length = struct.unpack("<I", frame[:4])[0]
    header = json.loads(frame[4:4 + length])
    offset = 4 + length
    offset += (-offset) % 4
    return header, np.frombuffer(frame[offset:], "<f4")


def test_headset_protocol_and_recording(roots, tmp_path):
    async def scenario():
        server = TeleopServer(ServerConfig(scene="cube", out_dir=tmp_path), roots)
        async with TestClient(TestServer(server.make_app())) as client:
            index = await client.get("/")
            assert "importmap" in await index.text()
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "hello", "role": "headset"})
            hello = await ws.receive_json(timeout=60)
            assert hello["type"] == "hello" and any(s["id"] == "cube" for s in hello["scenes"])

            server.runner.ready.wait(60)
            session = server.runner.session
            script = ScriptedGrasp("right", object_top=0.057, kinematics=session.hands["right"].kinematics)
            await ws.send_json({"type": "cmd", "cmd": "record_start"})
            geometry, frames, saved, t = None, 0, None, 0.0
            while saved is None:
                frame = script.frame(t)
                t += 1.0 / 60.0
                await ws.send_json({
                    "type": "hands",
                    "head": [0.0, 0.5, 0.45, 0.0, 0.0, 0.0, 1.0],
                    "right": {"tracked": True, "p": frame.joints.reshape(-1).tolist(),
                              "q": [0.0, 0.0, 0.0, 1.0] * 25, "r": [0.008] * 25},
                    "left": {"tracked": False},
                })
                msg = await ws.receive(timeout=30)
                if msg.type == WSMsgType.BINARY:
                    header, payload = _decode(msg.data)
                    expected = header["num_actors"] * 7 + 3 * sum(c for _, c in header["deformables"])
                    assert len(payload) == expected + 6 * header["num_contacts"]
                    frames += 1
                    if frames == 120:
                        assert header["tracked"]["right"] and not header["tracked"]["left"]
                        await ws.send_json({"type": "cmd", "cmd": "record_stop"})
                elif msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data["type"] == "geometry":
                        geometry = data
                    elif data.get("saved"):
                        saved = data["saved"]
                await asyncio.sleep(0.005)
            assert geometry is not None and len(geometry["actors"]) == 40
            await ws.close()
        return saved

    saved = asyncio.run(scenario())
    import h5py

    with h5py.File(saved) as f:
        assert f.attrs["num_steps"] > 60
        assert np.isfinite(f["head/pose"][-1]).all()
        assert f["hands/right/tracked"][-1]
