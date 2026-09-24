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

"""The streamed hand skeleton: 25 joints in WebXR / OpenXR order.

Meta Quest hand tracking exposes the same 25 joints through WebXR Hand Input
(``XRHand``) and OpenXR ``XR_EXT_hand_tracking`` (minus the OpenXR palm joint):
the wrist, four thumb joints, then five joints for each of the other fingers.
All positions are in meters, in the physics world frame (Y-up), after the
client has applied its workspace anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)
NUM_JOINTS = len(JOINT_NAMES)

WRIST = 0
THUMB_METACARPAL = 1
THUMB_PROXIMAL = 2
THUMB_DISTAL = 3
THUMB_TIP = 4
INDEX_METACARPAL = 5
MIDDLE_METACARPAL = 10
RING_METACARPAL = 15
PINKY_METACARPAL = 20

# Knuckle (MCP) joints used for the palm frame, and fingertips.
INDEX_PROXIMAL = INDEX_METACARPAL + 1
MIDDLE_PROXIMAL = MIDDLE_METACARPAL + 1
FINGER_TIPS = (THUMB_TIP, 9, 14, 19, 24)

SIDES = ("left", "right")


@dataclass
class HandFrame:
    """One tracked hand at one instant.

    Attributes:
        tracked: False when the headset lost this hand; joints are then stale.
        joints: (25, 3) joint positions [m] in the physics world frame.
        rotations: Optional (25, 4) world-from-joint quaternions [x, y, z, w]
            as reported by the runtime (recorded, not needed for retargeting).
        radii: Optional (25,) joint radii [m] as reported by the runtime.
    """

    tracked: bool = False
    joints: np.ndarray = field(default_factory=lambda: np.zeros((NUM_JOINTS, 3)))
    rotations: np.ndarray | None = None
    radii: np.ndarray | None = None

    @staticmethod
    def untracked() -> "HandFrame":
        return HandFrame(tracked=False)

    @staticmethod
    def from_flat(
        tracked: bool,
        positions: list[float] | np.ndarray,
        rotations: list[float] | np.ndarray | None = None,
        radii: list[float] | np.ndarray | None = None,
    ) -> "HandFrame":
        joints = np.asarray(positions, dtype=np.float64).reshape(NUM_JOINTS, 3)
        rot = (
            None
            if rotations is None or len(rotations) == 0
            else np.asarray(rotations, dtype=np.float64).reshape(NUM_JOINTS, 4)
        )
        rad = (
            None
            if radii is None or len(radii) == 0
            else np.asarray(radii, dtype=np.float64).reshape(NUM_JOINTS)
        )
        valid = bool(tracked) and bool(np.all(np.isfinite(joints)))
        return HandFrame(tracked=valid, joints=joints, rotations=rot, radii=rad)


def estimate_palm_frame(
    wrist: np.ndarray, index_knuckle: np.ndarray, middle_knuckle: np.ndarray
) -> np.ndarray | None:
    """Palm frame from [wrist, index knuckle, middle knuckle].

    Port of AnyDexRetarget's ``estimate_frame_from_hand_points`` (as used by
    the visionOS example): x runs from the middle knuckle toward the wrist
    projected off the palm normal, y is the palm-plane normal and z = x × y,
    with the normal disambiguated toward the index side. Applying the same
    estimator to the tracked hand and to the bot's rest skeleton makes left
    and right hands consistent without hand-written mirror conventions.

    Returns:
        (3, 3) rotation whose columns are the frame axes, or None when the
        three points are degenerate.
    """
    p1 = np.asarray(index_knuckle, dtype=np.float64) - wrist
    p2 = np.asarray(middle_knuckle, dtype=np.float64) - wrist
    x_vector = -p2
    normal = np.cross(p1, p2)
    n = np.linalg.norm(normal)
    if n < 1e-9:
        return None
    normal /= n
    x = x_vector - normal * np.dot(x_vector, normal)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x /= nx
    z = np.cross(x, normal)
    if np.dot(z, p1 - p2) < 0.0:
        normal = -normal
        z = -z
    return np.stack([x, normal, z], axis=1)
