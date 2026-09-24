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

"""Scenes, physical hands, and contact/force recording."""

import h5py
import numpy as np
import pytest

from superdex_quest_teleop import hand_skeleton as hs
from superdex_quest_teleop.scenes import build_scene, scene_registry
from superdex_quest_teleop.session import TeleopSession
from superdex_quest_teleop.synthetic import ScriptedGrasp

FAST_SCENES = ["sphere", "cube", "duck_lamp", "soft_cube", "free_rope", "paper_cup", "box_and_blocks", "medley"]


@pytest.mark.parametrize("scene_id", FAST_SCENES)
def test_scene_builds_and_settles(roots, scene_id):
    import superdex.physics as physics

    spec = scene_registry(roots)[scene_id]
    scene = build_scene(spec, roots)
    try:
        for _ in range(30):
            scene.step(spec.time_step)
        actors = []
        scene.for_each_actor(lambda a: actors.append(a))
        for actor in actors:
            if actor.get_name() != "table":
                box = actor.get_aabb_world()
                assert box.min[1] > -0.01, actor.get_name()  # nothing sank through the table
                assert abs(box.min[0]) < 0.6 and abs(box.min[2]) < 0.6
    finally:
        physics.destroy_scene(scene)


def test_all_prefabs_are_registered(roots):
    registry = scene_registry(roots)
    for name in ("box_and_blocks", "duck_lamp", "sphere", "shape_box", "chain", "paper_cup_pyramid",
                 "nine_hole_peg_test", "functional_dexterity_test", "cube", "free_rope", "cloth"):
        assert name in registry


def _run_grasp(roots, scene_id, tmp_path, seconds=6.0, **kwargs):
    session = TeleopSession(scene_registry(roots)[scene_id], roots, **kwargs)
    target = session.object_records[0]
    top = max(r.actor.get_aabb_world().max[1] for r in session.object_records)
    script = ScriptedGrasp("right", object_top=top, kinematics=session.hands["right"].kinematics)
    path = session.start_recording(tmp_path, {"test": True})
    heights = []
    for k in range(int(seconds / session.time_step)):
        session.set_input({"right": script.frame(k * session.time_step), "left": hs.HandFrame.untracked()})
        data = session.step()
        heights.append(data["pose"][target.index, 1])
    saved, steps = session.stop_recording()
    session.close()
    assert saved == path
    return path, steps, np.asarray(heights), target


def test_scripted_grasp_lifts_cube_and_records_contacts(roots, tmp_path):
    path, steps, heights, target = _run_grasp(roots, "cube", tmp_path)
    assert heights.max() > 0.05, "the scripted grasp should lift the cube off the table"
    with h5py.File(path) as f:
        assert f.attrs["num_steps"] == steps == len(f["steps/sim_time"])
        names = f["actors/name"][:].astype(str)
        owner, other = f["contacts/owner"][:], f["contacts/other"][:]
        offsets = f["contacts/step_offset"][:]
        assert offsets[-1] == len(owner) > 0
        side = f["actors/hand_side"][:].astype(str)
        # Hand mode: every recorded point involves a hand, never a hand touching itself.
        involves_hand = (side[owner] != "") | (side[np.maximum(other, 0)] != "")
        assert np.all(involves_hand)
        assert not np.any((side[owner] != "") & (side[owner] == side[np.maximum(other, 0)]) & (other >= 0))
        cube = int(np.where(names == target.name)[0][0])
        assert np.any((owner == cube) | (other == cube))
        assert f["hands/right/tracked"][:].all() and not f["hands/left/tracked"][:].any()
        assert f["hands/right/joints"].shape == (steps, 25, 3)
        assert f["hands/right/link_pose"].shape == (steps, 19, 7)


def test_contact_points_reconstruct_total_force(roots, tmp_path):
    """Per-point forces sum to the engine's total contact force per actor."""
    path, _, _, target = _run_grasp(roots, "cube", tmp_path, seconds=3.0, contact_mode="all")
    with h5py.File(path) as f:
        names = f["actors/name"][:].astype(str)
        cube = int(np.where(names == target.name)[0][0])
        owner, other = f["contacts/owner"][:], f["contacts/other"][:]
        force, offsets = f["contacts/force"][:], f["contacts/step_offset"][:]
        total = f["actors/contact_force"][:, cube]
        checked = 0
        for t in range(len(offsets) - 1):
            s = slice(offsets[t], offsets[t + 1])
            recon = force[s][owner[s] == cube].sum(0) - force[s][other[s] == cube].sum(0)
            if np.linalg.norm(total[t]) > 0.05:
                np.testing.assert_allclose(recon, total[t], atol=1e-3 + 1e-3 * np.linalg.norm(total[t]))
                checked += 1
        assert checked > 10


def test_deformable_node_forces_recorded(roots, tmp_path):
    path, steps, _, _ = _run_grasp(roots, "duck_lamp", tmp_path, seconds=3.0)
    with h5py.File(path) as f:
        group = f["deformables/0"]
        assert group["surface_positions"].shape[0] == steps
        assert group["node_force_offset"][-1] == len(group["node_index"]) > 0
        assert group["surface_triangles"].shape[1] == 3
