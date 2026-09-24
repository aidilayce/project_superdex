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

"""Physical Meta XR hands driven by Quest hand tracking.

Python port of the visionOS example's ``HandUnit`` (TeleopRig.cpp): one
dynamic articulated Meta XR hand bot per side, tracking an FK-consistent
set of link targets through SuperDex's articulated pose controller. The
wrist follows the tracked wrist with a stiff Cartesian spring, the fingers
follow the retargeted joint angles in joint space, and every link collides
with the scene, so hand/object forces come from the contact solver.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics

from . import hand_skeleton as hs
from .hand_model import BotKinematics, matrix_to_quat
from .retarget import MetaXrHandRetargeter, RetargetConfig, RetargetResult
from .scenes import AssetRoots

# Physical profile applied to both sides (see the visionOS HandUnit: the
# checked-in left/right assets differ by 98x in density and 100x in contact
# penalty, so both sides use the right hand's published values). A hand-side
# Coulomb coefficient of 2.0 combines geometrically with the engine default
# of 0.5 on props to an effective hand/prop friction of 1.0.
HAND_DENSITY = 980.0
HAND_PENALTY = 1e8
HAND_FRICTION = 2.0

# Pose controller gains (visionOS HandUnit at life scale).
WRIST_POS_GAINS = (2000.0, 60.0)
WRIST_ROT_GAINS = (50.0, 1.5)
FINGER_JOINT_GAINS = (0.5, 0.05, 0.6)  # stiffness, damping, saturation

SMOOTHER_FRAMES = 5


def hand_bot_path(roots: AssetRoots, side: str, variant: str = "lowpoly") -> str:
    return str(
        roots.assets / "bots" / "hands" / "oculus_xr" / side
        / f"oculus_xr_hand_{variant}_{side}.superdex_bot"
    )


def _to_transform(m: np.ndarray) -> physics.TransformRT:
    return physics.TransformRT(
        rotation=physics.Quaternion(*[float(v) for v in matrix_to_quat(m[:3, :3])]),
        translation=[float(v) for v in m[:3, 3]],
    )


def _gains(stiffness: float, damping: float, saturation: float = -1.0):
    return physics.PoseTrackingParams(
        stiffness=stiffness, damping=damping, saturation=saturation
    )


class HandUnit:
    """One physical Meta XR hand in a scene."""

    def __init__(
        self,
        scene: physics.Scene,
        roots: AssetRoots,
        side: str,
        spawn: tuple[float, float, float],
        variant: str = "lowpoly",
        retarget_config: RetargetConfig | None = None,
    ) -> None:
        self.side = side
        self.scene = scene
        prefab = robotics.load_bot_prefab_from_file(hand_bot_path(roots, side, variant))
        for link in prefab.links:
            link.has_gravity = False  # the pose controller has no gravity term
            link.density = HAND_DENSITY
            link.contact.penalty_coefficient = HAND_PENALTY
            link.contact.coulomb_friction_coefficient = HAND_FRICTION
        self.kinematics = BotKinematics(prefab)
        self.retargeter = MetaXrHandRetargeter(self.kinematics, side, retarget_config)

        # Neutral spawn: palm down, fingers pointing away from the operator.
        world_from_root = np.eye(4)
        world_from_root[:3, :3] = self._neutral_rotation()
        world_from_root[:3, 3] = spawn
        prefab.world_from_root = _to_transform(world_from_root)

        self.name = f"hand_{side}"
        prefab.name = self.name
        self._robotics = robotics.create_context()
        self.bot = robotics.create_bot(scene, prefab, self._robotics)
        self.actor = self.bot.get_articulated_actor()
        self.num_links = self.kinematics.num_links
        self.link_actors = [scene.get_actor(h) for h in self.actor.get_nested_link_actors()]
        self.link_names = [a.get_name() for a in self.link_actors]

        n = self.num_links
        link_pos = [physics.PoseTrackingParams() for _ in range(n)]
        link_rot = [physics.PoseTrackingParams() for _ in range(n)]
        joints = [physics.PoseTrackingParams() for _ in range(n)]
        link_pos[0] = _gains(*WRIST_POS_GAINS)
        link_rot[0] = _gains(*WRIST_ROT_GAINS)
        for i in range(1, n):
            kind = str(prefab.joints[i].type).rsplit(".", 1)[-1].upper()
            if kind in ("REVOLUTE", "SPHERICAL"):
                joints[i] = _gains(*FINGER_JOINT_GAINS)
        # PoseControllerParams exposes copies of its arrays: pass them whole.
        self.actor.add_articulated_pose_controller(
            physics.PoseControllerParams(
                link_pos_tracking=physics.DynamicArrayPoseTrackingParams(link_pos),
                link_rot_tracking=physics.DynamicArrayPoseTrackingParams(link_rot),
                joint_tracking=physics.DynamicArrayPoseTrackingParams(joints),
            )
        )

        self._frames: deque[np.ndarray] = deque(maxlen=SMOOTHER_FRAMES)
        self._placed = False
        self._reset_targets = True
        self.tracked = False
        self.last_input: hs.HandFrame = hs.HandFrame.untracked()
        self.last_result: RetargetResult | None = None

    def _neutral_rotation(self) -> np.ndarray:
        right = self.side == "right"
        wrist = np.zeros(3)
        middle = np.array([0.0, -0.02, -0.1])
        index = np.array([-0.03 if right else 0.03, -0.02, -0.1])
        palm = hs.estimate_palm_frame(wrist, index, middle)
        return palm @ self.retargeter._bot_palm.T

    # ------------------------------------------------------------------

    def _smooth(self, joints: np.ndarray) -> np.ndarray:
        """AnyDex MediaPipeSmoother: exp-weighted mean of the last 5 frames."""
        if self._frames:
            # A joint at exactly the origin is a runtime dropout: hold it.
            dropped = np.all(joints == 0.0, axis=1)
            if np.any(dropped):
                joints = joints.copy()
                joints[dropped] = self._frames[-1][dropped]
        else:
            for _ in range(SMOOTHER_FRAMES - 1):
                self._frames.append(joints)
        self._frames.append(joints)
        stack = np.stack(self._frames)
        weights = np.exp(np.linspace(-2.0, 0.0, len(stack)))
        return np.tensordot(weights / weights.sum(), stack, axes=1)

    def set_input(self, frame: hs.HandFrame) -> None:
        """Retarget the latest tracked hand and set the controller targets.

        Call once per physics step, before stepping."""
        self.last_input = frame
        if not frame.tracked:
            if self.tracked:
                # Do not infer a target velocity across the tracking gap and
                # never teleport on reacquisition (the hand may be touching
                # something): hold the last targets until tracking returns.
                self._frames.clear()
                self._reset_targets = True
                self.retargeter.reset()
            self.tracked = False
            return
        self.tracked = True
        result = self.retargeter.retarget(self._smooth(frame.joints))
        if result is None:
            return
        self.last_result = result
        targets = physics.DynamicArrayTransformRT(
            [_to_transform(m) for m in result.world_from_links]
        )
        if not self._placed:
            # First acquisition: move the idle hand onto the tracked one.
            self.actor.set_articulated_pose_from_links(targets)
            self.actor.set_articulated_joint_velocities(
                np.zeros(self.actor.get_num_dofs(), dtype=_real_dtype())
            )
            self._placed = True
            self._reset_targets = True
        if self._reset_targets:
            self.actor.reset_articulated_target_link_transforms(targets)
            self._reset_targets = False
        else:
            self.actor.set_articulated_target_link_transforms(targets)

    def link_poses(self) -> np.ndarray:
        """Current world poses of the links, (num_links, 7) [px py pz qx qy qz qw]."""
        out = physics.DynamicArrayTransformRT(self.num_links)
        self.actor.get_articulated_link_transforms(out)
        return np.array([list(t.translation) + list(t.rotation) for t in out])

    def target_qpos(self) -> np.ndarray:
        if self.last_result is None:
            return np.full(self.kinematics.num_dofs, np.nan)
        return self.last_result.qpos

    def destroy(self) -> None:
        robotics.destroy_bot(self.scene, self.bot)


def _real_dtype():
    return np.float64 if physics.uses_double_precision() else np.float32
