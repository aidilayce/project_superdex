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

"""Quest hand skeleton -> SuperDex Meta XR hand retargeting.

The Meta XR hand bot (``assets/bots/hands/oculus_xr``) is the OVR hand rig:
bone_00 wrist root, bone_02..05 thumb (trapezium, metacarpal, proximal,
distal), bone_06..08 index, bone_09..11 middle, bone_12..14 ring and
bone_15..18 pinky (with metacarpal). Each bone's origin sits on one of the
25 WebXR/OpenXR joints, so the tracked skeleton maps onto bot points
directly (see ``POINT_MAP``).

The solver follows the structure of the AnyDex retargeter used by the
visionOS example, reduced to what a closed-loop teleop rig needs:

1. The wrist orientation comes from the palm frame (wrist, index knuckle,
   middle knuckle) of both the tracked hand and the bot's rest skeleton.
2. Finger targets are the tracked bone *directions* rebuilt with the bot's
   own bone lengths ("full hand vectors" with per-wearer segment scaling),
   anchored at the bot's knuckles, so hands of any size map onto the fixed
   size bot without stretching.
3. A pinch term matches the thumb-to-fingertip vectors when the wearer's
   fingertips are close, so pinch grasps close on the bot too.
4. A damped Gauss-Newton solve with analytic Jacobians, joint-limit
   clamping, warm start and a velocity regularizer yields the 27 bot DoFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import hand_skeleton as hs
from .hand_model import BotKinematics

WRIST_LINK = "bone_00_wrist_root"

# (bot link, tracked joint) pairs whose link origin sits on the joint.
_ORIGIN_MAP: tuple[tuple[str, int], ...] = (
    ("bone_03_thumb_proximal", hs.THUMB_METACARPAL),
    ("bone_04_thumb_medial", hs.THUMB_PROXIMAL),
    ("bone_05_thumb_distal", hs.THUMB_DISTAL),
    ("bone_06_index_proximal", 6),
    ("bone_07_index_medial", 7),
    ("bone_08_index_distal", 8),
    ("bone_09_middle_proximal", 11),
    ("bone_10_middle_medial", 12),
    ("bone_11_middle_distal", 13),
    ("bone_12_ring_proximal", 16),
    ("bone_13_ring_medial", 17),
    ("bone_14_ring_distal", 18),
    ("bone_15_pinky_metacarpal", 20),
    ("bone_16_pinky_proximal", 21),
    ("bone_17_pinky_medial", 22),
    ("bone_18_pinky_distal", 23),
)

# Fingertips: the distal bones carry no tip link; these are the distal
# collision-mesh endpoints along the link's local Y axis (mirrored between
# the left and right assets), as measured for the visionOS example.
_TIP_LINKS = (
    "bone_05_thumb_distal",
    "bone_08_index_distal",
    "bone_11_middle_distal",
    "bone_14_ring_distal",
    "bone_18_pinky_distal",
)
_TIP_LENGTHS = {
    "right": (0.022060, 0.019071, 0.021590, 0.020753, 0.019140),
    "left": (-0.022072, -0.019293, -0.021616, -0.020718, -0.019307),
}

# Tracked joint chains per finger (thumb, index, middle, ring, pinky). The
# first entry of each chain is where the target chain is anchored.
_CHAINS: tuple[tuple[int, ...], ...] = (
    (hs.THUMB_METACARPAL, hs.THUMB_PROXIMAL, hs.THUMB_DISTAL, hs.THUMB_TIP),
    (6, 7, 8, 9),
    (11, 12, 13, 14),
    (16, 17, 18, 19),
    (20, 21, 22, 23, 24),
)


@dataclass
class RetargetConfig:
    """Tuning of the retargeting objective and solver."""

    # Residual weights: per chain point (knuckle ... tip).
    joint_weight: float = 1.0
    tip_weight: float = 2.0
    # Pinch term: active between d_far (alpha = 0) and d_near (alpha = 1) of
    # thumb-tip to finger-tip distance [m].
    pinch_weight: float = 3.0
    pinch_near: float = 0.02
    pinch_far: float = 0.045
    # Velocity regularization toward the previous solution.
    regularization: float = 1e-4
    # Levenberg damping and iteration counts (cold start / warm start).
    damping: float = 1e-4
    cold_iterations: int = 20
    warm_iterations: int = 4
    # Exponential low-pass on the output joint angles (1 = no filtering).
    output_alpha: float = 0.6
    per_finger_weights: tuple[float, ...] = field(
        default_factory=lambda: (1.0, 1.0, 1.0, 1.0, 1.0)
    )


@dataclass
class RetargetResult:
    """Output of one retargeting call (all in the physics world frame)."""

    qpos: np.ndarray  # (num_dofs,) bot DoFs
    world_from_root: np.ndarray  # 4x4 wrist target
    residual: float  # RMS target error [m]
    kinematics: BotKinematics | None = None

    @property
    def world_from_links(self) -> list[np.ndarray]:
        """4x4 per link, FK-consistent (computed on first use)."""
        cached = self.__dict__.get("_links")
        if cached is None:
            links, _ = self.kinematics.forward(self.qpos)
            cached = self.__dict__["_links"] = [self.world_from_root @ link for link in links]
        return cached


class MetaXrHandRetargeter:
    """Retargets a 25-joint tracked hand onto one Meta XR hand bot."""

    def __init__(
        self,
        kinematics: BotKinematics,
        side: str,
        config: RetargetConfig | None = None,
    ) -> None:
        if side not in hs.SIDES:
            raise ValueError(f"side must be one of {hs.SIDES}")
        self.kin = kinematics
        self.side = side
        self.config = config or RetargetConfig()
        k = kinematics

        # Tracked points: link origins, then fingertips.
        self._point_links: list[int] = []
        offsets: list[np.ndarray] = []
        self._point_joint: list[int] = []
        for link, joint in _ORIGIN_MAP:
            self._point_links.append(k.link_index(link))
            offsets.append(np.zeros(3))
            self._point_joint.append(joint)
        for f, link in enumerate(_TIP_LINKS):
            self._point_links.append(k.link_index(link))
            # Distal link -> fingertip, scaled with a calibrated (scaled) hand.
            offsets.append(np.array([0.0, _TIP_LENGTHS[side][f] * getattr(kinematics, "scale", 1.0), 0.0]))
            self._point_joint.append(hs.FINGER_TIPS[f])
        self._point_offsets = np.asarray(offsets)
        self._point_of_joint = {j: i for i, j in enumerate(self._point_joint)}

        q0 = np.zeros(k.num_dofs)
        rest, _, rest_links = k.points_and_jacobians(
            q0, self._point_links, self._point_offsets
        )
        self._rest_points = rest

        # Bot palm frame from its rest skeleton (wrist at the root origin).
        palm = hs.estimate_palm_frame(
            np.zeros(3),
            rest_links[k.link_index("bone_06_index_proximal")][:3, 3],
            rest_links[k.link_index("bone_09_middle_proximal")][:3, 3],
        )
        if palm is None:
            raise RuntimeError("degenerate bot palm frame")
        self._bot_palm = palm

        # Bot bone lengths along each chain.
        self._chain_lengths = []
        for chain in _CHAINS:
            pts = [rest[self._point_of_joint[j]] for j in chain]
            self._chain_lengths.append(
                [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
            )
        # Palm size (wrist -> middle knuckle) for scaling the thumb base.
        self._bot_palm_length = float(
            np.linalg.norm(rest[self._point_of_joint[11]])
        )

        weights = []
        for j in self._point_joint:
            finger = next(i for i, c in enumerate(_CHAINS) if j in c)
            w = self.config.tip_weight if j in hs.FINGER_TIPS else self.config.joint_weight
            weights.append(w * self.config.per_finger_weights[finger])
        self._weights = np.sqrt(np.asarray(weights))

        self._q_prev: np.ndarray | None = None
        self._q_filtered: np.ndarray | None = None

    # ------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        return self.kin.num_dofs

    def reset(self) -> None:
        self._q_prev = None
        self._q_filtered = None

    def wrist_rotation(self, joints: np.ndarray) -> np.ndarray | None:
        """World-from-root rotation that aligns the bot palm with the hand."""
        palm = hs.estimate_palm_frame(
            joints[hs.WRIST], joints[hs.INDEX_PROXIMAL], joints[hs.MIDDLE_PROXIMAL]
        )
        if palm is None:
            return None
        return palm @ self._bot_palm.T

    def targets(self, local: np.ndarray) -> np.ndarray:
        """Bot-frame target points from wrist-local tracked joints."""
        targets = self._rest_points.copy()
        palm_len = float(np.linalg.norm(local[11]))
        scale = self._bot_palm_length / palm_len if palm_len > 1e-4 else 1.0
        for chain, lengths in zip(_CHAINS, self._chain_lengths):
            first = self._point_of_joint[chain[0]]
            if chain[0] == hs.THUMB_METACARPAL:
                # The thumb base moves with the bot's trapezium joint; place it
                # by the wearer's palm-scaled CMC position.
                anchor = local[chain[0]] * scale
            else:
                # Knuckles are rigid on the bot: anchor at its own.
                anchor = self._rest_points[first]
            targets[first] = anchor
            for a, b, length in zip(chain[:-1], chain[1:], lengths):
                bone = local[b] - local[a]
                n = np.linalg.norm(bone)
                direction = bone / n if n > 1e-6 else np.zeros(3)
                anchor = anchor + direction * length
                targets[self._point_of_joint[b]] = anchor
        return targets

    def _pinch_terms(self, local: np.ndarray) -> list[tuple[int, int, float, np.ndarray]]:
        cfg = self.config
        terms = []
        thumb = local[hs.THUMB_TIP]
        for f in range(1, 5):
            tip = local[hs.FINGER_TIPS[f]]
            d = float(np.linalg.norm(tip - thumb))
            alpha = np.clip((cfg.pinch_far - d) / (cfg.pinch_far - cfg.pinch_near), 0.0, 1.0)
            if alpha > 0.0:
                terms.append(
                    (
                        self._point_of_joint[hs.THUMB_TIP],
                        self._point_of_joint[hs.FINGER_TIPS[f]],
                        float(alpha),
                        thumb - tip,
                    )
                )
        return terms

    def solve(self, local: np.ndarray) -> tuple[np.ndarray, float]:
        """Bot DoFs for wrist-local (bot root frame) tracked joints."""
        cfg = self.config
        k = self.kin
        targets = self.targets(local)
        pinch = self._pinch_terms(local)
        warm = self._q_prev is not None
        q = (
            self._q_prev.copy()
            if warm
            else np.clip(np.zeros(k.num_dofs), k.lower, k.upper)
        )
        q_reg = self._q_prev
        iterations = cfg.warm_iterations if warm else cfg.cold_iterations
        w = self._weights[:, None]
        residual = 0.0
        for _ in range(iterations):
            pts, jac, _ = k.points_and_jacobians(q, self._point_links, self._point_offsets)
            residual = float(np.sqrt(np.mean(np.sum((pts - targets) ** 2, axis=1))))
            r = ((pts - targets) * w).reshape(-1)
            J = (jac * w[:, :, None]).reshape(-1, k.num_dofs)
            rows_r = [r]
            rows_j = [J]
            for a, b, alpha, human in pinch:
                s = np.sqrt(cfg.pinch_weight * alpha)
                rows_r.append(s * ((pts[a] - pts[b]) - human))
                rows_j.append(s * (jac[a] - jac[b]))
            if q_reg is not None and cfg.regularization > 0.0:
                s = np.sqrt(cfg.regularization)
                rows_r.append(s * (q - q_reg))
                rows_j.append(s * np.eye(k.num_dofs))
            r_all = np.concatenate(rows_r)
            J_all = np.concatenate(rows_j, axis=0)
            H = J_all.T @ J_all + cfg.damping * np.eye(k.num_dofs)
            g = J_all.T @ r_all
            q = np.clip(q - np.linalg.solve(H, g), k.lower, k.upper)
        # The residual is the one before the last update (an upper bound;
        # re-evaluating it would cost another forward pass per frame).
        self._q_prev = q
        return q, residual

    def retarget(self, joints: np.ndarray) -> RetargetResult | None:
        """Full pipeline for world-frame tracked joints (25, 3).

        Returns None when the palm frame is degenerate (bad tracking).
        """
        rot = self.wrist_rotation(joints)
        if rot is None:
            return None
        world_from_root = np.eye(4)
        world_from_root[:3, :3] = rot
        world_from_root[:3, 3] = joints[hs.WRIST]
        local = (joints - joints[hs.WRIST]) @ rot  # rot^T (p - wrist), row form
        q, residual = self.solve(local)
        if self._q_filtered is None:
            self._q_filtered = q.copy()
        else:
            self._q_filtered += self.config.output_alpha * (q - self._q_filtered)
        return RetargetResult(
            qpos=self._q_filtered.copy(),
            world_from_root=world_from_root,
            residual=residual,
            kinematics=self.kin,
        )

    def synthesize_skeleton(
        self, q: np.ndarray, world_from_root: np.ndarray | None = None
    ) -> np.ndarray:
        """The 25-joint skeleton the bot itself would report at pose ``q``.

        Joints without a bot counterpart (wrist aside: the finger metacarpals
        of index/middle/ring) are interpolated between the wrist and the
        knuckle. Used by tests and by the synthetic hand source.
        """
        world_from_root = np.eye(4) if world_from_root is None else world_from_root
        pts, _, _ = self.kin.points_and_jacobians(q, self._point_links, self._point_offsets)
        out = np.zeros((hs.NUM_JOINTS, 3))
        for i, j in enumerate(self._point_joint):
            out[j] = pts[i]
        for meta in (hs.INDEX_METACARPAL, hs.MIDDLE_METACARPAL, hs.RING_METACARPAL):
            out[meta] = 0.35 * out[meta + 1]
        out = out @ world_from_root[:3, :3].T + world_from_root[:3, 3]
        return out
