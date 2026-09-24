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

"""Kinematics and retargeting of the Meta XR hand."""

import numpy as np
import pytest
import superdex.physics as physics
import superdex.robotics as robotics

from superdex_quest_teleop.hand_model import BotKinematics, rotvec_to_matrix, transform_to_matrix
from superdex_quest_teleop.hand_rig import hand_bot_path
from superdex_quest_teleop.retarget import MetaXrHandRetargeter, RetargetConfig

SIDES = ("left", "right")


def _prefab(roots, side):
    return robotics.load_bot_prefab_from_file(hand_bot_path(roots, side))


@pytest.mark.parametrize("side", SIDES)
def test_forward_kinematics_matches_engine(roots, side):
    prefab = _prefab(roots, side)
    kin = BotKinematics(prefab)
    assert kin.num_dofs == 27
    scene = physics.create_scene("fk")
    try:
        bot = robotics.create_bot(scene, prefab, robotics.create_context())
        actor = bot.get_articulated_actor()
        q = np.random.default_rng(3).uniform(kin.lower, kin.upper)
        pose = robotics.build_articulated_pose_from_bot_pose(
            prefab, q.astype(np.float32).tolist(), actor.get_num_dofs()
        )
        actor.set_articulated_pose_from_joints(np.asarray(pose, dtype=np.float32))
        scene.step(0.0)
        out = physics.DynamicArrayTransformRT(kin.num_links)
        actor.get_articulated_link_transforms(out)
        root = transform_to_matrix(out[0])
        links, _ = kin.forward(q)
        for i in range(kin.num_links):
            np.testing.assert_allclose(root @ links[i], transform_to_matrix(out[i]), atol=1e-5)
    finally:
        physics.destroy_scene(scene)


@pytest.mark.parametrize("side", SIDES)
def test_point_jacobians_match_finite_differences(roots, side):
    kin = BotKinematics(_prefab(roots, side))
    q = np.random.default_rng(5).uniform(kin.lower, kin.upper)
    links = [kin.link_index(n) for n in ("bone_05_thumb_distal", "bone_08_index_distal", "bone_18_pinky_distal")]
    offsets = np.array([[0.0, 0.02, 0.0]] * 3)
    pos, jac, _ = kin.points_and_jacobians(q, links, offsets)
    eps = 1e-6
    for d in range(kin.num_dofs):
        dq = q.copy()
        dq[d] += eps
        pos2, _, _ = kin.points_and_jacobians(dq, links, offsets)
        np.testing.assert_allclose((pos2 - pos) / eps, jac[:, :, d], atol=1e-4)


@pytest.mark.parametrize("side", SIDES)
def test_retarget_round_trip(roots, side):
    """A skeleton synthesized from a bot pose retargets back onto it."""
    kin = BotKinematics(_prefab(roots, side))
    retargeter = MetaXrHandRetargeter(kin, side, RetargetConfig(output_alpha=1.0))
    rng = np.random.default_rng(0)
    q = np.zeros(kin.num_dofs)
    errors = []
    for step in range(40):
        q = np.clip(q + rng.normal(0.0, 0.08, kin.num_dofs), 0.8 * kin.lower, 0.8 * kin.upper)
        world = np.eye(4)
        world[:3, :3] = rotvec_to_matrix(np.array([0.3, 1.0, -0.2]) * 0.05 * step)
        world[:3, 3] = [0.1, 0.3, -0.2]
        skeleton = retargeter.synthesize_skeleton(q, world)
        result = retargeter.retarget(skeleton)
        np.testing.assert_allclose(result.world_from_root, world, atol=1e-9)
        errors.append(np.abs(retargeter.synthesize_skeleton(result.qpos, result.world_from_root) - skeleton).max())
    assert np.median(errors) < 0.002  # [m]
    assert np.all(result.qpos >= kin.lower - 1e-9) and np.all(result.qpos <= kin.upper + 1e-9)


def test_retarget_is_hand_size_invariant(roots):
    """A bigger or smaller wearer produces the same bot pose (bone
    directions drive the fingers, the bot keeps its own proportions)."""
    kin = BotKinematics(_prefab(roots, "right"))
    q = np.clip(np.random.default_rng(1).normal(0.0, 0.3, kin.num_dofs), 0.7 * kin.lower, 0.7 * kin.upper)
    results = []
    for scale in (0.85, 1.0, 1.2):
        retargeter = MetaXrHandRetargeter(kin, "right", RetargetConfig(output_alpha=1.0, cold_iterations=40))
        skeleton = retargeter.synthesize_skeleton(q) * scale
        results.append(retargeter.synthesize_skeleton(retargeter.retarget(skeleton).qpos))
    for other in results[1:]:
        assert np.abs(other - results[0]).max() < 0.004
