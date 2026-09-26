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

"""Display skeleton of a *simulated* Meta XR hand in WebXR joint convention.

The client draws each hand as one continuous skinned mesh (the WebXR
``generic-hand`` model, or any rigged hand whose bones carry the 25 WebXR
joint names). Its bones take WebXR joint poses: -Z along the bone toward
the fingertip, +Y dorsal. This module turns the simulated bot's link poses
into those 25 poses, so the skin follows the physics (including where
contact stops a finger), not the raw tracking.

Rest orientations come from the skinned model's own bind pose, transferred
onto the bot's rest skeleton through the two palm frames; at runtime each
joint's dorsal axis rides on its link's rotation and its -Z axis is re-aimed
at the next joint.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import hand_skeleton as hs
from .hand_model import matrices_to_quats, quats_to_matrices
from .retarget import MetaXrHandRetargeter

GENERIC_HAND_DIR = Path(__file__).resolve().parent / "web" / "vendor" / "webxr-input-profiles" / "generic-hand"


def _glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    offset = 20 + length
    bin_length = struct.unpack("<I", data[offset:offset + 4])[0]
    return doc, data[offset + 8:offset + 8 + bin_length]


def bind_pose(path: Path) -> dict[str, np.ndarray]:
    """World-from-bone 4x4 bind matrices of a skinned GLB, by bone name."""
    doc, blob = _glb(path)
    skin = doc["skins"][0]
    accessor = doc["accessors"][skin["inverseBindMatrices"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    inverse = np.frombuffer(blob, np.float32, 16 * accessor["count"], start)
    inverse = inverse.reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
    return {
        doc["nodes"][node]["name"]: np.linalg.inv(inverse[i])
        for i, node in enumerate(skin["joints"])
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _next_joint(j: int) -> int:
    """The joint a bone points at (the wrist points at the middle knuckle)."""
    if j == hs.WRIST:
        return hs.MIDDLE_PROXIMAL
    return j + 1


class HandDisplayRig:
    """Maps simulated link poses to 25 WebXR-convention joint poses."""

    def __init__(self, retargeter: MetaXrHandRetargeter, model: Path | None = None) -> None:
        side = retargeter.side
        kin = retargeter.kin
        model = model or GENERIC_HAND_DIR / f"{side}.glb"
        bind = bind_pose(model)
        gen = np.array([bind[name][:3, 3] for name in hs.JOINT_NAMES])
        gen_rot = np.array([bind[name][:3, :3] for name in hs.JOINT_NAMES])

        q0 = np.zeros(kin.num_dofs)
        rest_links, _ = kin.forward(q0)
        # Bot rest skeleton (root frame); metacarpals without bot joints are
        # placed from the model, scaled by palm length.
        bot = retargeter.synthesize_skeleton(q0)
        bot_palm = hs.estimate_palm_frame(bot[0], bot[hs.INDEX_PROXIMAL], bot[hs.MIDDLE_PROXIMAL])
        gen_palm = hs.estimate_palm_frame(gen[0], gen[hs.INDEX_PROXIMAL], gen[hs.MIDDLE_PROXIMAL])
        transfer = bot_palm @ gen_palm.T  # model frame -> bot root frame
        scale = np.linalg.norm(bot[hs.MIDDLE_PROXIMAL]) / np.linalg.norm(gen[hs.MIDDLE_PROXIMAL] - gen[0])
        for meta in (hs.INDEX_METACARPAL, hs.MIDDLE_METACARPAL, hs.RING_METACARPAL):
            bot[meta] = transfer @ (gen[meta] - gen[0]) * scale

        # Each joint rides on one link: the link whose origin sits on it
        # (the bone leaving the joint), the distal link for fingertips and
        # the root for the wrist and the rigid metacarpals.
        link_of: dict[int, int] = {}
        for i, joint in enumerate(retargeter._point_joint):
            link_of[joint] = retargeter._point_links[i]
        root = kin.link_index("bone_00_wrist_root")
        for joint in (hs.WRIST, hs.INDEX_METACARPAL, hs.MIDDLE_METACARPAL, hs.RING_METACARPAL):
            link_of[joint] = root
        self.link = np.array([link_of[j] for j in range(hs.NUM_JOINTS)])
        # Joint positions and dorsal axes in their link's frame.
        self.local_pos = np.zeros((hs.NUM_JOINTS, 3))
        self.local_dorsal = np.zeros((hs.NUM_JOINTS, 3))
        for j in range(hs.NUM_JOINTS):
            link = rest_links[self.link[j]]
            self.local_pos[j] = link[:3, :3].T @ (bot[j] - link[:3, 3])
            # Model rest dorsal (+Y) moved onto the bot, off its bone axis.
            bone = _normalize(bot[_next_joint(j)] - bot[j]) if not self._is_tip(j) else _normalize(bot[j] - bot[j - 1])
            dorsal = transfer @ gen_rot[j][:, 1]
            dorsal = _normalize(dorsal - bone * np.dot(dorsal, bone))
            self.local_dorsal[j] = link[:3, :3].T @ dorsal
        self._tips = np.array([self._is_tip(j) for j in range(hs.NUM_JOINTS)])
        self._next = np.array([j if self._is_tip(j) else _next_joint(j) for j in range(hs.NUM_JOINTS)])
        self._prev = np.array([j - 1 if self._is_tip(j) else j for j in range(hs.NUM_JOINTS)])

    @staticmethod
    def _is_tip(j: int) -> bool:
        return j in hs.FINGER_TIPS

    def joints(self, link_poses: np.ndarray) -> np.ndarray:
        """(25, 7) joint poses [px py pz qx qy qz qw] from (L, 7) link poses."""
        rot = quats_to_matrices(link_poses[:, 3:])
        r = rot[self.link]
        pos = np.einsum("jab,jb->ja", r, self.local_pos) + link_poses[self.link, :3]
        # -Z toward the next joint (tips continue the last bone).
        forward = pos[self._next] - pos
        forward[self._tips] = pos[self._tips] - pos[self._prev[self._tips]]
        z = -_normalize(forward)
        y = np.einsum("jab,jb->ja", r, self.local_dorsal)
        y = _normalize(y - z * np.sum(y * z, axis=1, keepdims=True))
        x = np.cross(y, z)
        out = np.zeros((hs.NUM_JOINTS, 7))
        out[:, :3] = pos
        out[:, 3:] = matrices_to_quats(np.stack([x, y, z], axis=2))
        return out
