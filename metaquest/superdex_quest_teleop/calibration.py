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

"""Hand-size calibration.

When the operator enters VR they hold both hands open for a moment; the
client sends the median 25-joint skeleton of each open hand (Quest hand
tracking already fits the skeleton to the user's hand). The simulated Meta
XR hands are then scaled uniformly so their bones match: the ratio of the
operator's bone lengths to the bot's, over the palm and the four fingers
(the thumb's tracking is the least reliable), median over segments and
averaged over both hands (a person's hands are the same size, so this
halves the tracking noise).
"""

from __future__ import annotations

import numpy as np

from . import hand_skeleton as hs

MIN_SCALE, MAX_SCALE = 0.75, 1.35

# WebXR joint indices: each finger is metacarpal, proximal, intermediate,
# distal, tip (index 5-9, middle 10-14, ring 15-19, pinky 20-24).
_FINGERS = (hs.INDEX_METACARPAL, hs.MIDDLE_METACARPAL, hs.RING_METACARPAL, hs.PINKY_METACARPAL)
RING_PROXIMAL = hs.RING_METACARPAL + 1
PINKY_PROXIMAL = hs.PINKY_METACARPAL + 1
INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = (m + 4 for m in _FINGERS)

# (from, to) joint pairs whose lengths define the hand size.
SEGMENTS: tuple[tuple[int, int], ...] = (
    (hs.WRIST, hs.MIDDLE_PROXIMAL),  # palm length
    (hs.INDEX_PROXIMAL, PINKY_PROXIMAL),  # knuckle width
    (hs.INDEX_PROXIMAL, INDEX_TIP),
    (hs.MIDDLE_PROXIMAL, MIDDLE_TIP),
    (RING_PROXIMAL, RING_TIP),
    (PINKY_PROXIMAL, PINKY_TIP),
)


def segment_lengths(joints: np.ndarray) -> np.ndarray:
    """Lengths of SEGMENTS for one (25, 3) skeleton. Finger segments are
    measured along the bones (sum of phalanges), so slight flexion doesn't
    shorten them."""
    joints = np.asarray(joints, np.float64).reshape(hs.NUM_JOINTS, 3)
    out = []
    for a, b in SEGMENTS:
        if _same_finger(a, b):
            out.append(sum(np.linalg.norm(joints[j + 1] - joints[j]) for j in range(a, b)))
        else:
            out.append(np.linalg.norm(joints[b] - joints[a]))
    return np.asarray(out)


def _same_finger(a: int, b: int) -> bool:
    return any(m <= a <= m + 4 and m <= b <= m + 4 for m in _FINGERS)


def hand_length(joints: np.ndarray) -> float:
    """Wrist to middle fingertip along the bones [m] (the usual hand length)."""
    lengths = segment_lengths(joints)
    return float(lengths[0] + lengths[3])


def scale_factor(measured: np.ndarray, reference: np.ndarray) -> float:
    """Uniform scale taking ``reference`` (the bot) to ``measured`` (the user)."""
    m, r = segment_lengths(measured), segment_lengths(reference)
    ratios = m / np.maximum(r, 1e-6)
    ratios = ratios[np.isfinite(ratios)]
    if not len(ratios):
        return 1.0
    return float(np.clip(np.median(ratios), MIN_SCALE, MAX_SCALE))


def calibrate(measured: dict[str, np.ndarray], references: dict[str, np.ndarray]) -> tuple[float, dict]:
    """One scale for both hands from each measured side (missing sides are
    skipped). Returns (scale, report)."""
    per_side = {}
    for side, joints in measured.items():
        if side in references and joints is not None:
            joints = np.asarray(joints, np.float64).reshape(hs.NUM_JOINTS, 3)
            if np.all(np.isfinite(joints)):
                per_side[side] = scale_factor(joints, references[side])
    if not per_side:
        raise ValueError("no usable hand measurement")
    scale = float(np.clip(np.mean(list(per_side.values())), MIN_SCALE, MAX_SCALE))
    lengths = {
        side: round(hand_length(np.asarray(measured[side]).reshape(hs.NUM_JOINTS, 3)) * 100, 1)
        for side in per_side
    }
    return scale, {"per_side": per_side, "hand_length_cm": lengths}
