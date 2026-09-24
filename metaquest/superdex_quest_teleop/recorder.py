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

"""Episode recording: per-step state, per-point contacts and per-node
contact forces, written to one HDF5 file per episode.

Layout (T steps, N actors, K contact points in total, see README.md)::

    /                      attrs: format, scene_id, scene_name, time_step, ...
    /actors/name (N) kind (N) is_static (N) hand_side (N) mass (N)
    /steps/sim_time (T) wall_time (T) step (T) solver_ok (T)
    /actors/pose (T,N,7)  lin_vel (T,N,3)  ang_vel (T,N,3)
    /actors/contact_force (T,N,3)          total contact force per actor
    /hands/<side>/tracked (T) joints (T,25,3) joint_rotations (T,25,4)
                 joint_radii (T,25) target_qpos (T,27) link_pose (T,L,7)
                 retarget_residual (T)
    /head/pose (T,7)
    /contacts/step_offset (T+1)            CSR row pointer into K
    /contacts/owner (K) other (K)          actor indices (actor_a, actor_b)
    /contacts/pos_owner pos_other normal force vel_owner vel_other (K,3)
    /contacts/distance int_weight (K) element (K) sample (K) barycentric (K,3)
    /deformables/<i>/node_force_offset (T+1) node_index (M) node_force (M,3)
    /deformables/<i>/surface_positions (T,V,3) surface_triangles (F,3)
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import h5py
import numpy as np

FORMAT = "superdex_quest_teleop/1"
FLUSH_EVERY = 120

CONTACT_VEC_FIELDS = ("pos_owner", "pos_other", "normal", "force", "vel_owner", "vel_other")


class _Appendable:
    """A 1-D-growing dataset with an in-memory buffer."""

    def __init__(self, group: h5py.Group, name: str, shape: tuple, dtype) -> None:
        self.ds = group.create_dataset(
            name,
            shape=(0, *shape),
            maxshape=(None, *shape),
            dtype=dtype,
            chunks=(max(1, min(1024, 65536 // max(1, int(np.prod(shape) or 1)))), *shape),
            compression="gzip" if int(np.prod(shape) or 1) >= 16 else None,
            compression_opts=1 if int(np.prod(shape) or 1) >= 16 else None,
        )
        self.shape = shape
        self.dtype = dtype
        self.buf: list[np.ndarray] = []

    def append(self, value) -> None:
        self.buf.append(np.asarray(value, dtype=self.dtype).reshape(self.shape))

    def extend(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=self.dtype).reshape((-1, *self.shape))
        if len(values):
            self.buf.append(values)

    def flush(self) -> None:
        if not self.buf:
            return
        rows = [b if b.shape != self.shape else b[None] for b in self.buf]
        data = np.concatenate(rows, axis=0)
        start = self.ds.shape[0]
        self.ds.resize(start + len(data), axis=0)
        self.ds[start:] = data
        self.buf.clear()


class EpisodeRecorder:
    """Writes one teleoperation episode to ``<out_dir>/<scene>_<time>.h5``."""

    def __init__(
        self,
        out_dir: str | Path,
        scene_id: str,
        scene_name: str,
        time_step: float,
        actors: list[dict],
        hand_links: dict[str, int],
        hand_dofs: int,
        deformables: list[dict],
        metadata: dict | None = None,
    ) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = out / f"{scene_id}_{stamp}.h5"
        suffix = 1
        while self.path.exists():
            self.path = out / f"{scene_id}_{stamp}_{suffix}.h5"
            suffix += 1
        self.file = h5py.File(self.path, "w")
        f = self.file
        f.attrs["format"] = FORMAT
        f.attrs["scene_id"] = scene_id
        f.attrs["scene_name"] = scene_name
        f.attrs["time_step"] = time_step
        f.attrs["created"] = datetime.datetime.now().isoformat()
        f.attrs["metadata"] = json.dumps(metadata or {})

        n = len(actors)
        g = f.create_group("actors")
        str_dt = h5py.string_dtype()
        g.create_dataset("name", data=np.array([a["name"] for a in actors], dtype=object), dtype=str_dt)
        g.create_dataset("kind", data=np.array([a["kind"] for a in actors], dtype=object), dtype=str_dt)
        g.create_dataset("hand_side", data=np.array([a.get("hand_side", "") for a in actors], dtype=object), dtype=str_dt)
        g.create_dataset("is_static", data=np.array([a["is_static"] for a in actors], dtype=bool))
        g.create_dataset("mass", data=np.array([a.get("mass", np.nan) for a in actors], dtype=np.float64))

        self._series: dict[str, _Appendable] = {}

        def series(path: str, shape: tuple, dtype=np.float32) -> None:
            group_name, name = path.rsplit("/", 1)
            group = f.require_group(group_name)
            self._series[path] = _Appendable(group, name, shape, dtype)

        series("steps/sim_time", (), np.float64)
        series("steps/wall_time", (), np.float64)
        series("steps/step", (), np.int64)
        series("actors/pose", (n, 7))
        series("actors/lin_vel", (n, 3))
        series("actors/ang_vel", (n, 3))
        series("actors/contact_force", (n, 3))
        series("head/pose", (7,))
        for side, num_links in hand_links.items():
            series(f"hands/{side}/tracked", (), np.bool_)
            series(f"hands/{side}/joints", (25, 3))
            series(f"hands/{side}/joint_rotations", (25, 4))
            series(f"hands/{side}/joint_radii", (25,))
            series(f"hands/{side}/target_qpos", (hand_dofs,))
            series(f"hands/{side}/link_pose", (num_links, 7))
            series(f"hands/{side}/retarget_residual", ())

        series("contacts/owner", (), np.int32)
        series("contacts/other", (), np.int32)
        for name in CONTACT_VEC_FIELDS:
            series(f"contacts/{name}", (3,))
        series("contacts/barycentric", (3,))
        series("contacts/distance", ())
        series("contacts/int_weight", ())
        series("contacts/element", (), np.int32)
        series("contacts/sample", (), np.int32)
        series("contacts/step_offset", (), np.int64)
        self._contact_count = 0
        self._series["contacts/step_offset"].append(0)

        self._deformables = deformables
        for i, d in enumerate(deformables):
            grp = f.require_group(f"deformables/{i}")
            grp.attrs["name"] = d["name"]
            grp.attrs["actor_index"] = d["actor_index"]
            if d.get("triangles") is not None:
                grp.create_dataset("surface_triangles", data=np.asarray(d["triangles"], np.int32))
            series(f"deformables/{i}/surface_positions", (d["num_surface_nodes"], 3))
            series(f"deformables/{i}/node_index", (), np.int32)
            series(f"deformables/{i}/node_force", (3,))
            series(f"deformables/{i}/node_force_offset", (), np.int64)
            self._series[f"deformables/{i}/node_force_offset"].append(0)
        self._node_counts = [0] * len(deformables)
        self.num_steps = 0
        self.num_contacts = 0
        self._t0 = time.time()

    def write_step(self, data: dict) -> None:
        """Append one physics step (see TeleopSession.gather)."""
        s = self._series
        s["steps/sim_time"].append(data["sim_time"])
        s["steps/wall_time"].append(time.time() - self._t0)
        s["steps/step"].append(data["step"])
        s["actors/pose"].append(data["pose"])
        s["actors/lin_vel"].append(data["lin_vel"])
        s["actors/ang_vel"].append(data["ang_vel"])
        s["actors/contact_force"].append(data["contact_force"])
        s["head/pose"].append(data.get("head_pose", np.full(7, np.nan)))
        for side, hand in data["hands"].items():
            s[f"hands/{side}/tracked"].append(hand["tracked"])
            s[f"hands/{side}/joints"].append(hand["joints"])
            s[f"hands/{side}/joint_rotations"].append(hand["joint_rotations"])
            s[f"hands/{side}/joint_radii"].append(hand["joint_radii"])
            s[f"hands/{side}/target_qpos"].append(hand["target_qpos"])
            s[f"hands/{side}/link_pose"].append(hand["link_pose"])
            s[f"hands/{side}/retarget_residual"].append(hand["residual"])

        c = data["contacts"]
        k = len(c["owner"])
        if k:
            s["contacts/owner"].extend(c["owner"])
            s["contacts/other"].extend(c["other"])
            for name in CONTACT_VEC_FIELDS:
                s[f"contacts/{name}"].extend(c[name])
            s["contacts/barycentric"].extend(c["barycentric"])
            s["contacts/distance"].extend(c["distance"])
            s["contacts/int_weight"].extend(c["int_weight"])
            s["contacts/element"].extend(c["element"])
            s["contacts/sample"].extend(c["sample"])
        self._contact_count += k
        self.num_contacts += k
        s["contacts/step_offset"].append(self._contact_count)

        for i, d in enumerate(data["deformables"]):
            s[f"deformables/{i}/surface_positions"].append(d["surface_positions"])
            m = len(d["node_index"])
            if m:
                s[f"deformables/{i}/node_index"].extend(d["node_index"])
                s[f"deformables/{i}/node_force"].extend(d["node_force"])
            self._node_counts[i] += m
            s[f"deformables/{i}/node_force_offset"].append(self._node_counts[i])

        self.num_steps += 1
        if self.num_steps % FLUSH_EVERY == 0:
            self.flush()

    def flush(self) -> None:
        for series in self._series.values():
            series.flush()
        self.file.flush()

    def close(self) -> Path:
        self.flush()
        self.file.attrs["num_steps"] = self.num_steps
        self.file.attrs["num_contacts"] = self.num_contacts
        self.file.close()
        return self.path
