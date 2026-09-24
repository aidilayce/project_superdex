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

"""Scripted hand motion in the Quest skeleton format.

Used to exercise the full pipeline (retargeting, contact, recording,
streaming) without a headset: a hand reaches down over the table center,
closes, lifts, opens and repeats. It produces exactly the 25-joint frames a
Quest would stream, so everything downstream runs unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from . import hand_skeleton as hs
from .hand_model import BotKinematics, rotvec_to_matrix
from .retarget import MetaXrHandRetargeter


def grasp_pose(kin: BotKinematics, amount: float, thumb: float = 0.6) -> np.ndarray:
    """Bot DoFs with every finger flexed by ``amount`` (0 open, 1 fist).

    Flexion is the direction of each joint's larger limit (the hand's flexion
    range is much wider than its extension range); spherical joints only
    flex about their first axis.
    """
    q = np.zeros(kin.num_dofs)
    for i, name in enumerate(kin.dof_names):
        lo, hi = kin.lower[i], kin.upper[i]
        if name.endswith("/ry") or name.endswith("/rz"):
            continue
        flex = lo if abs(lo) > abs(hi) else hi
        scale = thumb if "thumb" in name else 1.0
        q[i] = amount * scale * flex
    return q


class ScriptedGrasp:
    """Reach-grasp-lift-release cycle over a point on the table."""

    def __init__(
        self,
        side: str = "right",
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        object_top: float = 0.05,
        period: float = 6.0,
        kinematics: BotKinematics | None = None,
    ) -> None:
        if kinematics is None:
            raise ValueError("kinematics of the target hand bot is required")
        self.side = side
        self.kin = kinematics
        self.retargeter = MetaXrHandRetargeter(kinematics, side)
        self.target = np.asarray(target, dtype=np.float64)
        self.object_top = object_top
        self.period = period
        wrist = np.zeros(3)
        middle = np.array([0.0, -0.035, -0.1])
        index = np.array([-0.03 if side == "right" else 0.03, -0.035, -0.1])
        palm = hs.estimate_palm_frame(wrist, index, middle)
        self._rotation = palm @ self.retargeter._bot_palm.T

    def frame(self, t: float) -> hs.HandFrame:
        phase = (t % self.period) / self.period
        # 0-0.3 descend, 0.3-0.45 close, 0.45-0.7 lift, 0.7-0.85 lower+open, rest hover.
        def smooth(a: float, b: float) -> float:
            x = min(max((phase - a) / (b - a), 0.0), 1.0)
            return x * x * (3.0 - 2.0 * x)

        hover = self.object_top + 0.15
        grasp_height = self.object_top + 0.035
        height = hover + (grasp_height - hover) * smooth(0.0, 0.3)
        height += 0.12 * smooth(0.45, 0.7) - 0.12 * smooth(0.7, 0.85)
        height += (hover - grasp_height) * smooth(0.85, 1.0)
        close = smooth(0.3, 0.45) - smooth(0.75, 0.85)
        q = grasp_pose(self.kin, 0.55 * close)
        world_from_root = np.eye(4)
        world_from_root[:3, :3] = self._rotation @ rotvec_to_matrix(
            np.array([0.0, 0.0, 0.05 * math.sin(2 * math.pi * t / self.period)])
        )
        # Put the palm (about 7 cm from the wrist along the fingers) over the target.
        palm_offset = world_from_root[:3, :3] @ np.array([0.0, 0.07, 0.0])
        world_from_root[:3, 3] = self.target + np.array([0.0, height, 0.0]) - palm_offset * np.array([1, 0, 1])
        joints = self.retargeter.synthesize_skeleton(q, world_from_root)
        return hs.HandFrame(tracked=True, joints=joints)
