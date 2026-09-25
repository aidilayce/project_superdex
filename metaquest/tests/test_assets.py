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

"""The Poly Haven asset pack and the BEDLAM hand builder."""

import http.server
import json
import threading
from collections import Counter

import numpy as np
import pytest

from superdex_quest_teleop import bedlam_hands as bh
from superdex_quest_teleop import hand_skeleton as hs
from superdex_quest_teleop.environment import fetch_pack, load_manifest
from superdex_quest_teleop.hand_display import HandDisplayRig, bind_pose


class _FakeApi(http.server.BaseHTTPRequestHandler):
    """Responses shaped like api.polyhaven.com (see its swagger.yml)."""

    def do_GET(self):  # noqa: N802
        base = f"http://127.0.0.1:{self.server.server_port}/dl"
        tex = lambda i: {  # noqa: E731
            "Diffuse": {"1k": {"jpg": {"url": f"{base}/{i}_diff_1k.jpg"}}},
            "nor_gl": {"1k": {"jpg": {"url": f"{base}/{i}_nor_gl_1k.jpg"}}},
            "Rough": {"1k": {"jpg": {"url": f"{base}/{i}_rough_1k.jpg"}}},
        }
        routes = {
            "/assets?t=hdris": {"blinds": {"name": "Blinds", "categories": ["indoor"], "tags": ["kitchen"]},
                                "lebombo": {"name": "Lebombo", "categories": ["indoor"], "tags": []}},
            "/files/blinds": {"hdri": {"2k": {"hdr": {"url": f"{base}/blinds_2k.hdr"}}}},
            "/files/lebombo": {"hdri": {"2k": {"hdr": {"url": f"{base}/lebombo_2k.hdr"}}}},
            "/assets?t=textures": {"marble_01": {"name": "Marble 01"}, "kitchen_wood": {"name": "Kitchen Wood"},
                                   "long_white_tiles": {}, "laminate_floor": {}, "white_plaster_02": {},
                                   "oak_wood_planks": {}, "metal_plate_02": {}},
            "/assets?t=models": {"food_lime_01": {"name": "Lime", "categories": ["food"]},
                                 "ceramic_mug": {"name": "Ceramic Mug", "categories": ["food", "kitchen"],
                                                 "tags": ["mug"]},
                                 "office_chair": {"name": "Chair", "categories": ["furniture"]}},
        }
        for i in ("marble_01", "kitchen_wood", "long_white_tiles", "laminate_floor", "white_plaster_02",
                  "oak_wood_planks", "metal_plate_02"):
            routes[f"/files/{i}"] = tex(i)
        for m in ("food_lime_01", "ceramic_mug"):
            routes[f"/files/{m}"] = {"gltf": {"1k": {"gltf": {
                "url": f"{base}/{m}_1k.gltf",
                "include": {f"textures/{m}_diff_1k.jpg": {"url": f"{base}/{m}_diff_1k.jpg"}}}}}}
        if self.path in routes:
            body = json.dumps(routes[self.path]).encode()
        elif self.path.startswith("/dl/"):
            body = b"data:" + self.path.encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_asset_pack_downloads_materials_hdris_and_props(tmp_path):
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        manifest = fetch_pack(tmp_path, api=f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
    assert manifest["hdris"]["kitchen_morning"]["id"] == "blinds"
    assert manifest["materials"]["countertop"]["id"] == "marble_01"
    for slot in ("countertop", "cabinet", "backsplash", "floor", "wall", "tabletop", "steel"):
        entry = manifest["materials"][slot]
        for key in ("albedo", "normal", "roughness"):
            assert (tmp_path / entry[key]).exists(), (slot, key)
    assert [p["id"] for p in manifest["props"]] == ["food_lime_01", "ceramic_mug"]
    assert (tmp_path / "models/ceramic_mug/textures/ceramic_mug_diff_1k.jpg").exists()
    assert load_manifest(tmp_path) == manifest


@pytest.fixture(scope="module")
def fake_model(tmp_path_factory):
    import fake_smplx

    path = fake_smplx.make(tmp_path_factory.mktemp("smplx") / "SMPLX_FAKE.npz")
    return bh.load_smplx(path)


def test_bedlam_hands_are_watertight_rigged_and_textured(fake_model, tmp_path, roots):
    from PIL import Image

    texture = tmp_path / "skin_f_test_ALB.png"
    Image.new("RGB", (512, 512), (226, 172, 145)).save(texture)
    assert bh.find_textures(tmp_path) == [texture]
    written = bh.build(fake_model, texture, tmp_path / "out")
    assert [p.name for p in written] == ["left.glb", "right.glb"]
    for side in hs.SIDES:
        hand = bh.extract_hand(fake_model, side)
        _, weld = np.unique(np.round(hand["positions"], 6), axis=0, return_inverse=True)
        edges = Counter()
        for tri in weld.reshape(-1)[hand["indices"]]:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges[tuple(sorted((a, b)))] += 1
        assert all(count == 2 for count in edges.values())  # closed wrist
        np.testing.assert_allclose(hand["weights"].sum(axis=1), 1.0, atol=1e-6)
        bind = bind_pose(tmp_path / "out" / f"{side}.glb")
        assert set(bind) == set(hs.JOINT_NAMES)
        # The display rig accepts it as a hand model.
        import superdex.robotics as robotics

        from superdex_quest_teleop.hand_model import BotKinematics
        from superdex_quest_teleop.hand_rig import hand_bot_path
        from superdex_quest_teleop.retarget import MetaXrHandRetargeter

        kin = BotKinematics(robotics.load_bot_prefab_from_file(hand_bot_path(roots, side)))
        HandDisplayRig(MetaXrHandRetargeter(kin, side), tmp_path / "out" / f"{side}.glb")
