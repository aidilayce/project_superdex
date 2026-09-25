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
        server = TeleopServer(ServerConfig(scene="cube", out_dir=tmp_path, environment_auto=False), roots)
        async with TestClient(TestServer(server.make_app())) as client:
            index = await client.get("/")
            html = await index.text()
            assert "importmap" in html and "/vendor/three/" in html  # three.js served locally
            assert '"httpsPort": 8443' in html
            # WebXR needs a secure page: plain-HTTP LAN visitors go to HTTPS.
            lan = await client.get("/", headers={"Host": "192.168.1.20:8080"}, allow_redirects=False)
            assert lan.status == 302 and lan.headers["Location"] == "https://192.168.1.20:8443/"
            for path in ("/static/app.js", "/static/kitchen.js", "/static/skin.js",
                         "/vendor/three/build/three.module.js",
                         "/vendor/three/examples/jsm/loaders/GLTFLoader.js",
                         "/vendor/three/examples/jsm/loaders/RGBELoader.js",
                         "/static/hands.js",
                         "/vendor/webxr-input-profiles/generic-hand/right.glb",
                         "/ibl/studio_small_08_1k.hdr",
                         "/prefab_assets/sphere/render/sphere.glb"):
                assert (await client.get(path)).status == 200, path
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
                    assert header["hand_joints"] == ["left", "right"]
                    assert len(payload) == expected + 6 * header["num_contacts"] + 2 * 25 * 7
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
            renders = [a["render"]["url"] for a in geometry["actors"] if a.get("render")]
            assert len(renders) == 38  # every link of both hands has a smooth mesh
            assert (await client.get(renders[0])).status == 200
            await ws.close()
        return saved

    saved = asyncio.run(scenario())
    import h5py

    with h5py.File(saved) as f:
        assert f.attrs["num_steps"] > 60
        assert np.allclose(f["head/pose"][-1], [0.0, 0.5, 0.45, 0.0, 0.0, 0.0, 1.0])
        assert np.isfinite(f["head/pose"][-1]).all()
        assert f["hands/right/tracked"][-1]


def test_self_signed_certificate_lists_addresses(tmp_path):
    import subprocess

    from superdex_quest_teleop.server import ensure_self_signed_cert

    cert, key = ensure_self_signed_cert(tmp_path, ["192.168.1.20"])
    assert cert.exists() and key.exists()
    text = subprocess.run(["openssl", "x509", "-in", str(cert), "-noout", "-text"],
                          capture_output=True, text=True).stdout
    if text:  # openssl CLI available
        assert "192.168.1.20" in text and "localhost" in text
