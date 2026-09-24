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

"""One teleoperation session: a SuperDex scene, two physical Meta XR hands,
contact/force queries and an optional episode recorder.

All methods except ``set_input`` must be called from the thread that owns
the scene (the server's physics thread).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import superdex.physics as physics

from . import hand_skeleton as hs
from .hand_rig import HandUnit
from .recorder import EpisodeRecorder
from .retarget import RetargetConfig
from .scenes import TABLE_NAME, AssetRoots, SceneSpec, build_scene

Q = physics.QueryType

# Where the idle hands wait (physics frame: +Z is toward the operator).
HAND_SPAWN = {"left": (-0.15, 0.2, 0.25), "right": (0.15, 0.2, 0.25)}

# "hand": every contact point involving a hand link (hand/object and
# hand/table); "all": additionally object/object and object/table contacts.
# Contact samples live on one actor's surface (actor_a), and which side
# owns them depends on the collider pair (e.g. a rigid link against a soft
# body is sampled on the soft body), so both hands and objects are queried.
CONTACT_MODES = ("hand", "all")


@dataclass
class ActorRecord:
    index: int
    actor: physics.Actor
    name: str
    kind: str  # RIGID, SOFT, SHELL, ROD, ...
    is_static: bool
    hand_side: str  # "left"/"right" for hand links, else ""
    deformable: bool
    mesh_kind: str  # "surface", "visual", "plane" or "none"
    mass: float


def _kind(actor: physics.Actor) -> str:
    return str(actor.get_type()).rsplit(".", 1)[-1]


def _vec(v) -> list[float]:
    return v.tolist()


class TeleopSession:
    """A running teleoperation scene."""

    def __init__(
        self,
        spec: SceneSpec,
        roots: AssetRoots,
        sides: tuple[str, ...] = hs.SIDES,
        contact_mode: str = "hand",
        min_contact_force: float = 0.0,
        keep_self_contacts: bool = False,
        hand_variant: str = "lowpoly",
        retarget_config: RetargetConfig | None = None,
    ) -> None:
        if contact_mode not in CONTACT_MODES:
            raise ValueError(f"contact_mode must be one of {CONTACT_MODES}")
        self.spec = spec
        self.roots = roots
        self.contact_mode = contact_mode
        self.min_contact_force = min_contact_force
        self.keep_self_contacts = keep_self_contacts
        self.time_step = spec.time_step
        self.scene = build_scene(spec, roots)
        self.hands: dict[str, HandUnit] = {
            side: HandUnit(
                self.scene, roots, side, HAND_SPAWN[side], hand_variant, retarget_config
            )
            for side in sides
        }
        self.step_count = 0
        self.sim_time = 0.0
        self._lock = threading.Lock()
        self._input = {side: hs.HandFrame.untracked() for side in sides}
        self._head_pose = np.full(7, np.nan)
        self.recorder: EpisodeRecorder | None = None
        self._collect_actors()
        self._register_queries()
        self.last_contacts: dict = _empty_contacts()

    # ------------------------------------------------------------------ setup

    def _collect_actors(self) -> None:
        link_side: dict[str, str] = {}
        for side, hand in self.hands.items():
            for name in hand.link_names:
                link_side[name] = side
        all_actors: list[physics.Actor] = []
        self.scene.for_each_actor(lambda a: all_actors.append(a))
        records: list[ActorRecord] = []
        for actor in sorted(all_actors, key=lambda a: a.get_name()):
            kind = _kind(actor)
            if kind == "ARTICULATED":
                continue  # its nested link actors are listed individually
            name = actor.get_name()
            deformable = kind != "RIGID"
            mesh_kind = "none"
            if name == TABLE_NAME:
                mesh_kind = "plane"
            elif deformable and not actor.get_visual_mesh().is_empty():
                mesh_kind = "visual"
            elif not actor.get_surface_mesh().is_empty():
                mesh_kind = "surface"
            static = actor.is_static()
            mass = float("nan")
            if not static:
                try:
                    mass = float(actor.get_mass())
                except Exception:  # noqa: BLE001 - not every actor type reports mass
                    pass
            records.append(
                ActorRecord(
                    len(records), actor, name, kind, static, link_side.get(name, ""),
                    deformable, mesh_kind, mass,
                )
            )
        # Hands last so object indices are stable across hand configurations.
        records.sort(key=lambda r: (r.hand_side != "", r.hand_side, r.name))
        for i, r in enumerate(records):
            r.index = i
        self.actors = records
        self._index_of_handle = {r.actor.get_handle().value: r.index for r in records}
        self.hand_link_records = [r for r in records if r.hand_side]
        self._side_of = [r.hand_side for r in records] + [""]  # [-1] -> unknown
        self.object_records = [r for r in records if not r.hand_side and not r.is_static]
        self.deformables = [r for r in records if r.deformable]

    def _register_queries(self) -> None:
        self._queries = []

        def reg(actor: physics.Actor, query) -> bool:
            if actor.is_query_supported(query):
                self._queries.append(actor.register_query(query))
                return True
            return False

        self._has_total_force = {}
        for r in self.actors:
            if not r.is_static:
                self._has_total_force[r.index] = reg(r.actor, Q.TOTAL_CONTACT_FORCE)
        contact_sources = self.hand_link_records + self.object_records
        self._contact_sources = [r for r in contact_sources if reg(r.actor, Q.CONTACT_POINTS)]
        self._node_force = {}
        for r in self.deformables:
            self._node_force[r.index] = reg(r.actor, Q.NODE_CONTACT_FORCES)
            reg(r.actor, Q.VISUAL_NODE_POSITIONS if r.mesh_kind == "visual" else Q.SURFACE_NODE_POSITIONS)
        # Queries become valid after one step; a zero step does not advance time.
        self.scene.step(0.0)

    # ------------------------------------------------------------------ input

    def set_input(
        self,
        hands: dict[str, hs.HandFrame],
        head_pose: np.ndarray | None = None,
    ) -> None:
        """Latest tracking (thread-safe; the newest input wins)."""
        with self._lock:
            for side, frame in hands.items():
                if side in self._input:
                    self._input[side] = frame
            if head_pose is not None:
                self._head_pose = np.asarray(head_pose, dtype=np.float64)

    # ------------------------------------------------------------------ step

    def step(self, with_contacts: bool = True) -> dict:
        """Apply the latest hand input, advance one fixed step, gather data
        (and record it when recording). Returns the gathered step data.

        Reading per-point contacts costs a few microseconds per point, so
        callers may skip it on steps that are neither recorded nor displayed
        (recorded steps always include them)."""
        with self._lock:
            inputs = dict(self._input)
            head = self._head_pose.copy()
        for side, hand in self.hands.items():
            hand.set_input(inputs[side])
        self.scene.step(self.time_step)
        self.step_count += 1
        self.sim_time += self.time_step
        data = self.gather(head, with_contacts or self.recorder is not None)
        if self.recorder is not None:
            self.recorder.write_step(data)
        return data

    def gather(self, head_pose: np.ndarray | None = None, with_contacts: bool = True) -> dict:
        n = len(self.actors)
        pose = np.zeros((n, 7), np.float32)
        lin = np.full((n, 3), np.nan, np.float32)
        ang = np.full((n, 3), np.nan, np.float32)
        force = np.full((n, 3), np.nan, np.float32)
        for r in self.actors:
            a = r.actor
            if a.has_root_transform():
                t = a.get_root_transform()
                pose[r.index, :3] = _vec(t.translation)
                pose[r.index, 3:] = _vec(t.rotation)
            else:
                pose[r.index, 6] = 1.0
            if r.is_static:
                continue
            if not r.deformable:
                lin[r.index] = _vec(a.get_linear_velocity())
                ang[r.index] = _vec(a.get_angular_velocity())
            if self._has_total_force.get(r.index):
                force[r.index] = _vec(a.get_contact_force_world())

        contacts = self._gather_contacts() if with_contacts else None
        if contacts is not None:
            self.last_contacts = contacts

        deformables = []
        for r in self.deformables:
            a = r.actor
            if r.mesh_kind == "visual":
                local = np.asarray(a.get_visual_mesh_node_positions_local(), np.float32)
            else:
                local = np.asarray(a.get_surface_mesh_node_positions_local(), np.float32)
            world = _local_to_world(local.reshape(-1, 3), pose[r.index])
            idx = np.zeros(0, np.int32)
            nforce = np.zeros((0, 3), np.float32)
            if self._node_force.get(r.index):
                nf = a.get_node_contact_forces_world()
                if len(nf):
                    idx = np.fromiter((f.index for f in nf), np.int32, len(nf))
                    nforce = np.array([_vec(f.force) for f in nf], np.float32)
            deformables.append(
                {"surface_positions": world, "node_index": idx, "node_force": nforce}
            )

        hands = {}
        for side, hand in self.hands.items():
            frame = hand.last_input
            hands[side] = {
                "tracked": hand.tracked,
                "joints": frame.joints,
                "joint_rotations": frame.rotations if frame.rotations is not None
                else np.full((hs.NUM_JOINTS, 4), np.nan),
                "joint_radii": frame.radii if frame.radii is not None
                else np.full(hs.NUM_JOINTS, np.nan),
                "target_qpos": hand.target_qpos(),
                "link_pose": hand.link_poses(),
                "residual": hand.last_result.residual if hand.last_result else np.nan,
            }
        return {
            "step": self.step_count,
            "sim_time": self.sim_time,
            "pose": pose,
            "lin_vel": lin,
            "ang_vel": ang,
            "contact_force": force,
            "contacts": contacts,
            "deformables": deformables,
            "hands": hands,
            "head_pose": head_pose if head_pose is not None else np.full(7, np.nan),
        }

    def _gather_contacts(self) -> dict:
        rows: dict[str, list] = {k: [] for k in _empty_contacts()}
        threshold = self.min_contact_force
        index_of = self._index_of_handle
        side_of = self._side_of
        hand_only = self.contact_mode == "hand"
        keep_self = self.keep_self_contacts
        for r in self._contact_sources:
            owner_side = r.hand_side
            for c in r.actor.get_contact_points_world():
                other = index_of.get(c.actor_b.value, -1)
                other_side = side_of[other]
                if other == r.index and not r.deformable:
                    continue  # rigid self pairs carry no information
                if owner_side and other_side == owner_side and not keep_self:
                    continue  # the hand touching itself
                if hand_only and not owner_side and not other_side:
                    continue
                f = c.force.tolist()
                if threshold > 0.0 and (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) < threshold**2:
                    continue
                rows["owner"].append(r.index)
                rows["other"].append(other)
                rows["force"].append(f)
                rows["pos_owner"].append(c.pos_a.tolist())
                rows["pos_other"].append(c.pos_b.tolist())
                rows["normal"].append(c.normal.tolist())
                rows["vel_owner"].append(c.point_velocity_a.tolist())
                rows["vel_other"].append(c.point_velocity_b.tolist())
                rows["barycentric"].append(c.parametric_coords.tolist())
                rows["distance"].append(c.distance)
                rows["int_weight"].append(c.int_weight)
                rows["element"].append(c.element_index)
                rows["sample"].append(c.sample_index)
        out = {}
        for key, values in rows.items():
            template = _EMPTY[key]
            out[key] = np.asarray(values, template.dtype).reshape((-1, *template.shape[1:]))
        return out

    # ------------------------------------------------------------------ display

    def geometry(self) -> list[dict]:
        """Display meshes, one per actor, in actor-local coordinates (world
        coordinates for deformables, whose vertices stream every step)."""
        out = []
        for r in self.actors:
            entry = {
                "index": r.index, "name": r.name, "kind": r.kind, "static": r.is_static,
                "hand": r.hand_side, "deformable": r.deformable, "mesh": r.mesh_kind,
            }
            if r.hand_side:
                model = self.hands[r.hand_side].render_models.get(r.name)
                if model:
                    # The GLBs are Y-up exports of the Z-up link frames:
                    # link-frame vertices are the GLB's rotated +90 deg about X.
                    entry["render"] = {
                        "url": f"/hand_assets/{model}",
                        "rotation": [float(np.sin(np.pi / 4)), 0.0, 0.0, float(np.cos(np.pi / 4))],
                    }
            if r.mesh_kind in ("surface", "visual"):
                view = r.actor.get_visual_mesh() if r.mesh_kind == "visual" else r.actor.get_surface_mesh()
                coords = np.asarray(view.coordinates, np.float32)
                conn = np.asarray(view.connectivity, np.int64)
                if view.nodes_per_element == 3:
                    entry["positions"] = coords
                    entry["indices"] = conn.astype(np.uint32)
                else:
                    entry["mesh"] = "none"
            out.append(entry)
        return out

    def deformable_vertices(self, data: dict) -> list[np.ndarray]:
        return [d["surface_positions"] for d in data["deformables"]]

    # ------------------------------------------------------------------ recording

    def start_recording(self, out_dir: str | Path, metadata: dict | None = None) -> Path:
        if self.recorder is not None:
            return self.recorder.path
        deformables = []
        for r in self.deformables:
            view = r.actor.get_visual_mesh() if r.mesh_kind == "visual" else r.actor.get_surface_mesh()
            triangles = None
            if view.nodes_per_element == 3:
                triangles = np.asarray(view.connectivity, np.int64).reshape(-1, 3)
            deformables.append(
                {
                    "name": r.name, "actor_index": r.index, "triangles": triangles,
                    "num_surface_nodes": view.get_num_nodes(),
                }
            )
        hand = next(iter(self.hands.values()), None)
        meta = {
            "contact_mode": self.contact_mode,
            "min_contact_force": self.min_contact_force,
            "keep_self_contacts": self.keep_self_contacts,
            "precision": physics.PRECISION_NAME,
            "hand_bot": "oculus_xr (Meta XR Hand)",
            "hand_link_names": {s: h.link_names for s, h in self.hands.items()},
            "hand_dof_names": hand.kinematics.dof_names if hand else [],
            "joint_names": list(hs.JOINT_NAMES),
            "scene_description": self.spec.description,
        }
        meta.update(metadata or {})
        self.recorder = EpisodeRecorder(
            out_dir,
            self.spec.id,
            self.spec.name,
            self.time_step,
            [
                {"name": r.name, "kind": r.kind, "is_static": r.is_static,
                 "hand_side": r.hand_side, "mass": r.mass}
                for r in self.actors
            ],
            {s: h.num_links for s, h in self.hands.items()},
            hand.kinematics.num_dofs if hand else 0,
            deformables,
            meta,
        )
        return self.recorder.path

    def stop_recording(self) -> tuple[Path, int] | None:
        if self.recorder is None:
            return None
        rec = self.recorder
        self.recorder = None
        path = rec.close()
        return path, rec.num_steps

    def close(self) -> None:
        self.stop_recording()
        for hand in self.hands.values():
            hand.destroy()
        physics.destroy_scene(self.scene)


_EMPTY = {
    "owner": np.zeros((0,), np.int32),
    "other": np.zeros((0,), np.int32),
    "force": np.zeros((0, 3), np.float32),
    "pos_owner": np.zeros((0, 3), np.float32),
    "pos_other": np.zeros((0, 3), np.float32),
    "normal": np.zeros((0, 3), np.float32),
    "vel_owner": np.zeros((0, 3), np.float32),
    "vel_other": np.zeros((0, 3), np.float32),
    "barycentric": np.zeros((0, 3), np.float32),
    "distance": np.zeros((0,), np.float32),
    "int_weight": np.zeros((0,), np.float32),
    "element": np.zeros((0,), np.int32),
    "sample": np.zeros((0,), np.int32),
}


def _empty_contacts() -> dict:
    return {k: v.copy() for k, v in _EMPTY.items()}


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row vectors v (M,3) by quaternion q [x,y,z,w]."""
    u = q[:3]
    w = q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _local_to_world(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (_quat_rotate(pose[3:].astype(np.float64), local) + pose[:3]).astype(np.float32)
