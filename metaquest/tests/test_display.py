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

"""Display assets: skinned-hand joint poses, prefab render models, HDRIs."""

import http.server
import json
import threading

import numpy as np
import pytest
import superdex.robotics as robotics

from superdex_quest_teleop import hand_skeleton as hs
from superdex_quest_teleop.environment import fetch_kitchen_hdri, rank_hdris
from superdex_quest_teleop.hand_display import GENERIC_HAND_DIR, HandDisplayRig, bind_pose
from superdex_quest_teleop.hand_model import BotKinematics, matrix_to_quat, quat_to_matrix, rotvec_to_matrix
from superdex_quest_teleop.hand_rig import hand_bot_path
from superdex_quest_teleop.retarget import MetaXrHandRetargeter
from superdex_quest_teleop.scenes import prefab_render_models, scene_registry
from superdex_quest_teleop.session import TeleopSession
from superdex_quest_teleop.synthetic import grasp_pose


@pytest.mark.parametrize("side", hs.SIDES)
def test_display_joints_follow_webxr_convention(roots, side):
    kin = BotKinematics(robotics.load_bot_prefab_from_file(hand_bot_path(roots, side)))
    retargeter = MetaXrHandRetargeter(kin, side)
    rig = HandDisplayRig(retargeter)
    bind = bind_pose(GENERIC_HAND_DIR / f"{side}.glb")
    gen = np.array([bind[n][:3, 3] for n in hs.JOINT_NAMES])
    for amount in (0.0, 0.6):
        q = grasp_pose(kin, amount)
        world = np.eye(4)
        world[:3, :3] = rotvec_to_matrix(np.array([0.2, 1.1, 0.3]))
        world[:3, 3] = [0.1, 0.2, 0.3]
        links, _ = kin.forward(q)
        poses = np.array([np.r_[(world @ m)[:3, 3], matrix_to_quat((world @ m)[:3, :3])] for m in links])
        joints = rig.joints(poses)
        skeleton = retargeter.synthesize_skeleton(q, world)
        known = [j for j in range(25) if j not in (5, 10, 15)]
        np.testing.assert_allclose(joints[known, :3], skeleton[known], atol=1e-6)
        for j in range(25):
            r = quat_to_matrix(joints[j, 3:])
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-6)
            if j not in hs.FINGER_TIPS:
                nxt = hs.MIDDLE_PROXIMAL if j == 0 else j + 1
                bone = joints[nxt, :3] - joints[j, :3]
                assert -r[:, 2] @ bone / np.linalg.norm(bone) > 0.999  # -Z along the bone
        if amount == 0.0:
            # At rest every joint's dorsal axis matches the model's bind pose
            # relative to its own palm (the thumb is rotated off the palm).
            def palm_relation(p, rotations):
                ref = np.cross(p[21] - p[0], p[6] - p[0])
                ref /= np.linalg.norm(ref)
                return np.array([rot[:, 1] @ ref for rot in rotations])
            ours = palm_relation(joints[:, :3], [quat_to_matrix(x) for x in joints[:, 3:]])
            theirs = palm_relation(gen, [bind[n][:3, :3] for n in hs.JOINT_NAMES])
            assert np.abs(ours - theirs).max() < 0.1


def test_prefab_render_models_cover_scene_actors(roots):
    prefab = roots.assets / "prefabs" / "box_and_blocks" / "box_and_blocks.mochi_prefab"
    models = prefab_render_models(prefab, "prefab")
    assert models["prefab/Box"].name == "box.glb"
    assert models["prefab/Block_green7/Block_green"].name == "block_green.glb"
    session = TeleopSession(scene_registry(roots)["box_and_blocks"], roots)
    try:
        entries = {a["name"]: a for a in session.geometry()}
        rendered = [n for n, a in entries.items() if "render" in a and not a["hand"]]
        assert len(rendered) == 33  # the box and all 32 blocks
        assert entries["prefab/Box"]["render"]["url"] == "/prefab_assets/box_and_blocks/render/box.glb"
    finally:
        session.close()


class _FakePolyHaven(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        base = f"http://127.0.0.1:{self.server.server_port}"
        routes = {
            "/assets?t=hdris": {
                "sunny_meadow": {"name": "Sunny Meadow", "categories": ["outdoor"], "tags": []},
                "old_hall": {"name": "Old Hall", "categories": ["indoor"], "tags": ["hall"]},
                "cozy_kitchen": {"name": "Cozy Kitchen", "categories": ["indoor"], "tags": ["kitchen"]},
                "loft_apartment": {"name": "Loft", "categories": ["indoor"], "tags": ["apartment"]},
            },
            "/files/cozy_kitchen": {"hdri": {"2k": {"hdr": {"url": f"{base}/dl/cozy_kitchen_2k.hdr"}}}},
        }
        if self.path in routes:
            body = json.dumps(routes[self.path]).encode()
        elif self.path == "/dl/cozy_kitchen_2k.hdr":
            body = b"#?RADIANCE\n" + b"\0" * 64
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_kitchen_hdri_is_ranked_downloaded_and_cached(tmp_path):
    assets = {
        "a": {"name": "Kitchen", "categories": ["indoor"], "tags": []},
        "b": {"name": "Living room", "categories": ["indoor"], "tags": ["apartment"]},
        "c": {"name": "Beach", "categories": ["outdoor"], "tags": ["kitchen"]},
    }
    assert [asset for _, asset in rank_hdris(assets)] == ["a", "b"]
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakePolyHaven)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        api = f"http://127.0.0.1:{server.server_port}"
        path = fetch_kitchen_hdri(tmp_path, api=api)
        assert path.name == "cozy_kitchen_2k.hdr" and path.read_bytes().startswith(b"#?RADIANCE")
        assert json.loads((tmp_path / "cozy_kitchen.json").read_text())["license"] == "CC0 1.0"
        server.shutdown()
        assert fetch_kitchen_hdri(tmp_path, api=api) == path  # cached, no network
    finally:
        server.server_close()
