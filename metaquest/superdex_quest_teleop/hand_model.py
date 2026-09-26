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

"""Kinematics of a SuperDex bot prefab in numpy (forward kinematics and
point Jacobians), used by the retargeter.

The DoF layout is the bot's own (the root joint's DoFs excluded): revolute
joints contribute one angle about their axis, spherical joints a rotation
vector in the joint frame. This matches ``superdex::robotics::
ComputeLinkTransformsFromPose`` and the visionOS example's
``SuperdexHandModel`` adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    """[x, y, z, w] quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n < 1e-12 else 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to an [x, y, z, w] quaternion (w >= 0)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
             (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
             (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """Vectorized :func:`quat_to_matrix` for (N, 4) [x, y, z, w] quaternions."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.einsum("ni,ni->n", q, q)
    s = np.where(n < 1e-12, 0.0, 2.0 / np.where(n < 1e-12, 1.0, n))
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - s * (y * y + z * z)
    m[:, 0, 1] = s * (x * y - z * w)
    m[:, 0, 2] = s * (x * z + y * w)
    m[:, 1, 0] = s * (x * y + z * w)
    m[:, 1, 1] = 1 - s * (x * x + z * z)
    m[:, 1, 2] = s * (y * z - x * w)
    m[:, 2, 0] = s * (x * z - y * w)
    m[:, 2, 1] = s * (y * z + x * w)
    m[:, 2, 2] = 1 - s * (x * x + y * y)
    return m


def matrices_to_quats(m: np.ndarray) -> np.ndarray:
    """Vectorized :func:`matrix_to_quat` for (N, 3, 3) rotations."""
    m = np.asarray(m, dtype=np.float64)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    trace = m00 + m11 + m22
    # The four branches of matrix_to_quat, each with its own pivot.
    cands = np.empty((4, len(m), 4))
    s = np.sqrt(np.maximum(trace + 1.0, 1e-12)) * 2.0
    cands[0] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s,
                         (m[:, 1, 0] - m[:, 0, 1]) / s, 0.25 * s], -1)
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
    cands[1] = np.stack([0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s,
                         (m[:, 2, 1] - m[:, 1, 2]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
    cands[2] = np.stack([(m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s,
                         (m[:, 0, 2] - m[:, 2, 0]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
    cands[3] = np.stack([(m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s,
                         (m[:, 1, 0] - m[:, 0, 1]) / s], -1)
    branch = np.where(trace > 0.0, 0, np.where((m00 > m11) & (m00 > m22), 1, np.where(m11 > m22, 2, 3)))
    q = cands[branch, np.arange(len(m))]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.where(q[:, 3:4] < 0.0, -q, q)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotvec_to_matrix(r: np.ndarray) -> np.ndarray:
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    theta = math.sqrt(x * x + y * y + z * z)
    if theta < 1e-9:
        return np.array([[1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0]])
    x, y, z = x / theta, y / theta, z / theta
    s, c = math.sin(theta), math.cos(theta)
    t = 1.0 - c
    return np.array(
        [
            [c + t * x * x, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, c + t * z * z],
        ]
    )


def _rotvecs_to_matrices(r: np.ndarray) -> np.ndarray:
    """Vectorized :func:`rotvec_to_matrix` for (N, 3) rotation vectors."""
    theta = np.linalg.norm(r, axis=1)
    small = theta < 1e-9
    safe = np.where(small, 1.0, theta)
    k = r / safe[:, None]
    s, c = np.sin(theta), np.cos(theta)
    t = 1.0 - c
    x, y, z = k[:, 0], k[:, 1], k[:, 2]
    m = np.empty((len(r), 3, 3))
    m[:, 0, 0] = c + t * x * x
    m[:, 0, 1] = t * x * y - s * z
    m[:, 0, 2] = t * x * z + s * y
    m[:, 1, 0] = t * x * y + s * z
    m[:, 1, 1] = c + t * y * y
    m[:, 1, 2] = t * y * z - s * x
    m[:, 2, 0] = t * x * z - s * y
    m[:, 2, 1] = t * y * z + s * x
    m[:, 2, 2] = c + t * z * z
    if np.any(small):
        rx, ry, rz = r[small, 0], r[small, 1], r[small, 2]
        one = np.ones_like(rx)
        m[small] = np.stack([np.stack([one, -rz, ry], -1), np.stack([rz, one, -rx], -1),
                             np.stack([-ry, rx, one], -1)], 1)
    return m


def rotvec_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, angle in [0, pi]) of a rotation matrix."""
    q = matrix_to_quat(m)  # [x, y, z, w]
    if q[3] < 0.0:
        q = -q
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-12:
        return 2.0 * q[:3]
    return q[:3] * (2.0 * math.atan2(s, float(q[3])) / s)


def right_jacobian_so3(r: np.ndarray) -> np.ndarray:
    theta2 = float(np.dot(r, r))
    if theta2 < 1e-10:
        a = 0.5 - theta2 / 24.0
        b = 1.0 / 6.0 - theta2 / 120.0
    else:
        theta = np.sqrt(theta2)
        a = (1.0 - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
    k = _skew(r)
    return np.eye(3) - a * k + b * (k @ k)


def _right_jacobians_so3(r: np.ndarray) -> np.ndarray:
    """Vectorized :func:`right_jacobian_so3` for (N, 3) rotation vectors."""
    theta2 = np.einsum("ni,ni->n", r, r)
    small = theta2 < 1e-10
    theta = np.sqrt(np.where(small, 1.0, theta2))
    t2 = np.where(small, 1.0, theta2)
    a = np.where(small, 0.5 - theta2 / 24.0, (1.0 - np.cos(theta)) / t2)
    b = np.where(small, 1.0 / 6.0 - theta2 / 120.0, (theta - np.sin(theta)) / (t2 * theta))
    k = np.zeros((len(r), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -r[:, 2], r[:, 1]
    k[:, 1, 0], k[:, 1, 2] = r[:, 2], -r[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -r[:, 1], r[:, 0]
    return np.eye(3) - a[:, None, None] * k + b[:, None, None] * (k @ k)


def transform_to_matrix(t) -> np.ndarray:
    """superdex.physics.TransformRT (or None) to a 4x4 matrix."""
    m = np.eye(4)
    if t is None:
        return m
    m[:3, :3] = quat_to_matrix(list(t.rotation))
    m[:3, 3] = list(t.translation)
    return m


REVOLUTE, SPHERICAL, FIXED = 1, 3, 0
_EYE4 = np.eye(4)


@dataclass
class _Joint:
    kind: int
    dof: int  # first bot DoF, -1 for fixed
    axis: np.ndarray  # revolute axis in the joint frame (unit)


class BotKinematics:
    """Numpy FK/Jacobians for a tree-structured SuperDex bot (root fixed)."""

    def __init__(self, bot_prefab) -> None:
        links = list(bot_prefab.links)
        joints = list(bot_prefab.joints)
        if len(links) != len(joints):
            raise ValueError("bot prefab has mismatched links and joints")
        self.link_names = [str(link.name) for link in links]
        self.parents = [int(link.parent_link) for link in links]
        self.num_links = len(links)
        self._parent_from_joint = [
            transform_to_matrix(j.parent_link_from_joint) for j in joints
        ]
        self._joint_from_link = [
            transform_to_matrix(link.parent_joint_from_link) for link in links
        ]
        self._joints: list[_Joint] = []
        lower: list[float] = []
        upper: list[float] = []
        self.dof_names: list[str] = []
        dof = 0
        for i, j in enumerate(joints):
            kind_name = str(j.type).rsplit(".", 1)[-1].upper()
            has_limits = j.min_limit is not None and j.max_limit is not None
            if kind_name == "REVOLUTE":
                axis = np.asarray(list(j.axis), dtype=np.float64)
                axis /= np.linalg.norm(axis)
                self._joints.append(_Joint(REVOLUTE, dof, axis))
                lo, hi = -np.pi, np.pi
                if has_limits:
                    # Revolute limits are authored as axis-scaled vectors.
                    lo = float(np.dot(list(j.min_limit), axis))
                    hi = float(np.dot(list(j.max_limit), axis))
                lower.append(min(lo, hi))
                upper.append(max(lo, hi))
                self.dof_names.append(str(j.name))
                dof += 1
            elif kind_name == "SPHERICAL":
                self._joints.append(_Joint(SPHERICAL, dof, np.zeros(3)))
                for a, suffix in enumerate(("rx", "ry", "rz")):
                    lo, hi = -np.pi, np.pi
                    if has_limits:
                        lo = float(list(j.min_limit)[a])
                        hi = float(list(j.max_limit)[a])
                    lower.append(min(lo, hi))
                    upper.append(max(lo, hi))
                    self.dof_names.append(f"{j.name}/{suffix}")
                dof += 3
            elif kind_name in ("FREE", "HARD", "FIXED"):
                if kind_name == "FREE" and i != 0:
                    raise ValueError("only a free root joint is supported")
                self._joints.append(_Joint(FIXED, -1, np.zeros(3)))
            else:
                raise ValueError(f"unsupported joint type {kind_name}")
        self.num_dofs = dof
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self._index = {name: i for i, name in enumerate(self.link_names)}
        self._prepare_batched()

    def _prepare_batched(self) -> None:
        """Arrays for the vectorized forward pass: joint rotation sources and
        links grouped by tree depth (parents before children)."""
        self._parent_from_joint_arr = np.stack(self._parent_from_joint)
        self._joint_from_link_arr = np.stack(self._joint_from_link)
        self._parents_arr = np.asarray(self.parents)
        rev = [i for i, j in enumerate(self._joints) if j.kind == REVOLUTE and self.parents[i] >= 0]
        sph = [i for i, j in enumerate(self._joints) if j.kind == SPHERICAL and self.parents[i] >= 0]
        self._rev_links = np.asarray(rev, dtype=np.int64)
        self._rev_dofs = np.asarray([self._joints[i].dof for i in rev], dtype=np.int64)
        self._rev_axes = np.asarray([self._joints[i].axis for i in rev]).reshape(-1, 3)
        self._sph_links = np.asarray(sph, dtype=np.int64)
        self._sph_dofs = np.asarray([np.arange(self._joints[i].dof, self._joints[i].dof + 3) for i in sph],
                                    dtype=np.int64).reshape(-1, 3)
        depth = []
        for i, parent in enumerate(self.parents):
            depth.append(0 if parent < 0 else depth[parent] + 1)
        self._levels = [np.asarray([i for i, d in enumerate(depth) if d == level], dtype=np.int64)
                        for level in range(max(depth) + 1)]

    def link_index(self, name: str) -> int:
        return self._index[name]

    def forward(self, q: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Root-from-link 4x4 transforms for every link at bot pose ``q``.

        Returns:
            (root_from_links, root_from_joints_after): the second list holds
            each joint frame after its own rotation, which is where Jacobian
            axes are expressed.
        """
        links, after = self._forward_arrays(q)
        return list(links), list(after)

    def _forward_arrays(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`forward` as (num_links, 4, 4) arrays."""
        q = np.asarray(q, dtype=np.float64)
        # All joint rotations at once (Rodrigues), then the tree level by level.
        rotvec = np.zeros((self.num_links, 3))
        if len(self._rev_links):
            rotvec[self._rev_links] = self._rev_axes * q[self._rev_dofs, None]
        if len(self._sph_links):
            rotvec[self._sph_links] = q[self._sph_dofs]
        local = self._parent_from_joint_arr.copy()
        local[:, :3, :3] = local[:, :3, :3] @ _rotvecs_to_matrices(rotvec)
        after_arr = np.empty((self.num_links, 4, 4))
        links_arr = np.empty((self.num_links, 4, 4))
        roots = self._levels[0]
        after_arr[roots] = _EYE4  # the root link sits at the root frame (free joint at zero)
        links_arr[roots] = self._joint_from_link_arr[roots]
        for level in self._levels[1:]:
            after_arr[level] = links_arr[self._parents_arr[level]] @ local[level]
            links_arr[level] = after_arr[level] @ self._joint_from_link_arr[level]
        return links_arr, after_arr

    def points_and_jacobians(
        self,
        q: np.ndarray,
        point_links: list[int],
        point_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Root-frame positions (P, 3) and Jacobians (P, 3, num_dofs) of
        points rigidly attached to links (``point_offsets`` in link frames)."""
        q = np.asarray(q, dtype=np.float64)
        links_arr, after = self._forward_arrays(q)
        sel = links_arr[np.asarray(point_links)]
        positions = np.einsum("pij,pj->pi", sel[:, :3, :3], point_offsets) + sel[:, :3, 3]
        # One world-frame rotation axis and pivot per DoF.
        axes = np.zeros((self.num_dofs, 3))
        pivots = np.zeros((self.num_dofs, 3))
        if len(self._rev_links):
            a = after[self._rev_links]
            axes[self._rev_dofs] = np.einsum("nij,nj->ni", a[:, :3, :3], self._rev_axes)
            pivots[self._rev_dofs] = a[:, :3, 3]
        if len(self._sph_links):
            a = after[self._sph_links]
            jr = _right_jacobians_so3(q[self._sph_dofs])
            # Columns of R @ Jr are the axes of the three DoFs.
            axes[self._sph_dofs] = np.transpose(a[:, :3, :3] @ jr, (0, 2, 1))
            pivots[self._sph_dofs] = a[:, None, :3, 3]
        links = list(links_arr)
        mask = self._ancestor_mask(tuple(point_links))
        jac = np.cross(axes[None, :, :], positions[:, None, :] - pivots[None, :, :])
        jac *= mask[:, :, None]
        return positions, np.transpose(jac, (0, 2, 1)), links

    def _ancestor_mask(self, point_links: tuple[int, ...]) -> np.ndarray:
        """(P, num_dofs) 1 where the DoF moves the point's link."""
        cache = self.__dict__.setdefault("_mask_cache", {})
        mask = cache.get(point_links)
        if mask is None:
            mask = np.zeros((len(point_links), self.num_dofs))
            for p, link in enumerate(point_links):
                cursor = link
                while cursor > 0:
                    joint = self._joints[cursor]
                    if joint.kind == REVOLUTE:
                        mask[p, joint.dof] = 1.0
                    elif joint.kind == SPHERICAL:
                        mask[p, joint.dof:joint.dof + 3] = 1.0
                    cursor = self.parents[cursor]
            cache[point_links] = mask
        return mask
