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
from pathlib import Path

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics

from . import hand_skeleton as hs
from .hand_display import HandDisplayRig
from .hand_model import BotKinematics, matrix_to_quat, quat_to_matrix, rotvec_from_matrix, rotvec_to_matrix
from .retarget import MetaXrHandRetargeter, RetargetConfig, RetargetResult
from .scenes import AssetRoots

# Physical profile applied to both sides (see the visionOS HandUnit: the
# checked-in left/right assets differ by 98x in density and 100x in contact
# penalty, so both sides use the right hand's published values). The pair
# friction is the geometric mean of both sides' coefficients: 1.0 on the hand
# gives 0.7 against the engine default of 0.5 and 0.95 against a 0.9 sponge,
# in the range of skin on plastic/foam (2.0 made objects cling to fingers).
HAND_DENSITY = 980.0
HAND_PENALTY = 1e8
HAND_FRICTION = 1.0

# Pose controller gains: (stiffness, damping, saturation). The wrist spring
# is not saturated: SuperDex's saturated tracking lets a hand pressed into the
# counter sink through it (instead, see the unstick logic below).
WRIST_POS_GAINS = (2000.0, 60.0)
WRIST_ROT_GAINS = (50.0, 1.5)
# Finger joints [N m/rad]: up to 0.64 N m per joint, about twice the visionOS
# gains (0.5/0.6) for a firmer squeeze. Much stiffer fingers overpower small
# objects (a 3 cm sphere squirts out of a closing grasp).
FINGER_JOINT_GAINS = (0.8, 0.06, 0.8)

# Target rate limits: real hands rarely exceed these, tracking glitches
# (a hand jumping 20 cm in one frame, fingers flipping) do. Unlimited jumps
# yank the pose controller and make the solver report explosions.
MAX_WRIST_SPEED = 4.0  # [m/s]
MAX_WRIST_ANGULAR_SPEED = 25.0  # [rad/s]
MAX_JOINT_SPEED = 30.0  # [rad/s]

# Unstick: when the simulated hand stays far from the tracked one (caught on
# or in an object, or pushed deep into the counter), it stops colliding with
# the scene until it has caught up, then collides again.
UNSTICK_DISTANCE = 0.12  # [m] wrist error that counts as stuck
UNSTICK_ANGLE = 1.2  # [rad]
UNSTICK_AFTER = 0.5  # [s] continuously stuck before passing through
RESTICK_DISTANCE = 0.03  # [m] back within this: collide again
RESTICK_ANGLE = 0.3  # [rad]
# ...and only once no link is buried this deep in the static environment
# (a hand that caught up inside the counter would be shoved out violently).
RESTICK_MAX_BURY = 0.01  # [m]

SMOOTHER_FRAMES = 5


def hand_bot_path(roots: AssetRoots, side: str, variant: str = "lowpoly") -> str:
    return str(
        roots.assets / "bots" / "hands" / "oculus_xr" / side
        / f"oculus_xr_hand_{variant}_{side}.superdex_bot"
    )


def _scaled(t: physics.TransformRT, s: float) -> physics.TransformRT:
    return physics.TransformRT(rotation=t.rotation, translation=[float(v) * s for v in t.translation])


def scale_bot_prefab(prefab, scale: float) -> None:
    """Uniformly scale a bot prefab in place: collision shapes, render
    models and joint offsets (mass follows from the density)."""
    if scale == 1.0:
        return
    for link in prefab.links:
        link.shape_scale = [float(v) * scale for v in link.shape_scale]
        link.shape_translation = [float(v) * scale for v in link.shape_translation]
        link.render_model_scale = [float(v) * scale for v in link.render_model_scale]
        link.render_model_translation = [float(v) * scale for v in link.render_model_translation]
        link.parent_joint_from_link = _scaled(link.parent_joint_from_link, scale)
        if link.center_of_mass is not None:
            link.center_of_mass = [float(v) * scale for v in link.center_of_mass]
    for joint in prefab.joints:
        joint.parent_link_from_joint = _scaled(joint.parent_link_from_joint, scale)


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
        display_model: Path | None = None,
        time_step: float = 1.0 / 60.0,
        scale: float = 1.0,
    ) -> None:
        self.side = side
        self.scene = scene
        self.time_step = time_step
        prefab = robotics.load_bot_prefab_from_file(hand_bot_path(roots, side, variant))
        for link in prefab.links:
            link.has_gravity = False  # the pose controller has no gravity term
            link.density = HAND_DENSITY
            link.contact.penalty_coefficient = HAND_PENALTY
            link.contact.coulomb_friction_coefficient = HAND_FRICTION
        # Hand-size calibration: the operator's hand size (see calibration.py).
        self.scale = float(scale)
        scale_bot_prefab(prefab, self.scale)
        self.kinematics = BotKinematics(prefab)
        self.kinematics.scale = self.scale
        self.retargeter = MetaXrHandRetargeter(self.kinematics, side, retarget_config)
        # 25 WebXR-convention joint poses for the skinned display hand.
        self.display = HandDisplayRig(self.retargeter, display_model)

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
        # Smooth render meshes (GLB, relative to the oculus_xr asset dir) per
        # link actor name, for display only.
        hand_dir = roots.assets / "bots" / "hands" / "oculus_xr"
        self.render_models: dict[str, str] = {}
        for link in prefab.links:
            path = str(link.render_model_file or "")
            if path:
                try:
                    rel = Path(path).resolve().relative_to(hand_dir.resolve())
                except ValueError:
                    continue
                self.render_models[f"{self.name}/{link.name}"] = rel.as_posix()

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

        self._cmd_root: np.ndarray | None = None  # rate-limited wrist target
        self._cmd_q: np.ndarray | None = None  # rate-limited joint targets
        self.passing_through = False  # unstick mode: no contact with the scene
        self._stuck_time = 0.0
        self._scene_actors: list[physics.ActorHandle] = []
        self._static_boxes = np.zeros((0, 2, 3))  # (lo, hi) of static colliders
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
        root, q = self._rate_limit(result.world_from_root, result.qpos)
        links, _ = self.kinematics.forward(q)
        targets = physics.DynamicArrayTransformRT([_to_transform(root @ m) for m in links])
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

    def _rate_limit(self, root: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Move the commanded wrist pose and joint angles toward the retargeted
        ones by at most the speed limits per step (the first time: jump)."""
        if self._cmd_root is None or not self._placed:
            self._cmd_root, self._cmd_q = root.copy(), q.copy()
            return self._cmd_root, self._cmd_q
        dt = self.time_step
        out = self._cmd_root.copy()
        step = root[:3, 3] - out[:3, 3]
        dist = float(np.linalg.norm(step))
        limit = MAX_WRIST_SPEED * dt
        out[:3, 3] += step if dist <= limit else step * (limit / dist)
        rel = rotvec_from_matrix(out[:3, :3].T @ root[:3, :3])
        angle = float(np.linalg.norm(rel))
        limit = MAX_WRIST_ANGULAR_SPEED * dt
        if angle > limit:
            rel *= limit / angle
        out[:3, :3] = out[:3, :3] @ rotvec_to_matrix(rel)
        dq = np.clip(q - self._cmd_q, -MAX_JOINT_SPEED * dt, MAX_JOINT_SPEED * dt)
        self._cmd_root, self._cmd_q = out, self._cmd_q + dq
        return self._cmd_root, self._cmd_q

    # ------------------------------------------------------------------ unstick

    def set_scene_actors(self, handles: list[physics.ActorHandle], static_boxes=None) -> None:
        """Actors the hand stops touching while it passes through (everything
        but the hands), and the bounds (lo, hi) of the static box colliders it
        must be out of before it collides again."""
        self._scene_actors = list(handles)
        if static_boxes is not None and len(static_boxes):
            self._static_boxes = np.asarray(static_boxes, np.float64).reshape(-1, 2, 3)

    def _buried(self) -> float:
        """Deepest overlap [m] of a link's bounds with a static box."""
        if not len(self._static_boxes):
            return 0.0
        lo = np.array([list(a.get_aabb_world().min) for a in self.link_actors])[:, None]
        hi = np.array([list(a.get_aabb_world().max) for a in self.link_actors])[:, None]
        overlap = np.minimum(hi, self._static_boxes[None, :, 1]) - np.maximum(lo, self._static_boxes[None, :, 0])
        return float(max(np.max(np.min(overlap, axis=-1)), 0.0))

    def update_unstick(self) -> None:
        """Call after each step: switch contact off while the simulated hand is
        stuck far from the tracked one, and back on once it has caught up."""
        if self._cmd_root is None or not self.tracked:
            self._stuck_time = 0.0
            return
        wrist = self.link_poses()[0]
        err = float(np.linalg.norm(wrist[:3] - self._cmd_root[:3, 3]))
        rot = quat_to_matrix(wrist[3:])
        ang = float(np.linalg.norm(rotvec_from_matrix(rot.T @ self._cmd_root[:3, :3])))
        if not self.passing_through:
            stuck = err > UNSTICK_DISTANCE or ang > UNSTICK_ANGLE
            self._stuck_time = self._stuck_time + self.time_step if stuck else 0.0
            if self._stuck_time >= UNSTICK_AFTER:
                self._set_scene_contact(False)
        elif err < RESTICK_DISTANCE and ang < RESTICK_ANGLE and self._buried() < RESTICK_MAX_BURY:
            self._set_scene_contact(True)
            self._stuck_time = 0.0

    def _set_scene_contact(self, enable: bool) -> None:
        handle = self.actor.get_handle()
        for other in self._scene_actors:
            self.scene.enable_actor_contact_symmetric(
                handle, other, enable, physics.IncludeNestedActors.YES
            )
        self.passing_through = not enable

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
