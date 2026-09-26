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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import superdex.physics as physics

from . import hand_skeleton as hs
from .hand_rig import HandUnit
from .recorder import EpisodeRecorder
from .retarget import RetargetConfig
from .scenes import TABLE_NAME, AssetRoots, SceneSpec, build_scene, forget_scene, workspace_for
from .workspace import DEFAULT_COUNTER_HEIGHT, is_environment_actor

Q = physics.QueryType

# Where the idle hands wait (physics frame: +Z is toward the operator).
HAND_SPAWN = {"left": (-0.15, 0.2, 0.25), "right": (0.15, 0.2, 0.25)}

# "hand": every contact point involving a hand link (hand/object and
# hand/table); "all": additionally object/object and object/table contacts.
# Contact samples live on one actor's surface (actor_a), and which side
# owns them depends on the collider pair (e.g. a rigid link against a soft
# body is sampled on the soft body), so both hands and objects are queried.
CONTACT_MODES = ("hand", "all")
# The solver always gets at least this share of the step period.
MIN_SOLVER_SHARE = 0.35


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


# Deformables stream (and record) their visual mesh every step unless it is
# huge: the t-shirt's subdivided visual mesh has 57k vertices (0.7 MB per
# frame) over a 3.6k-node simulation mesh, which then streams instead.
MAX_VISUAL_NODES = 20000


def _usable_visual_mesh(actor: physics.Actor) -> bool:
    visual = actor.get_visual_mesh()
    if visual.is_empty() or visual.nodes_per_element != 3:
        return False
    surface = actor.get_surface_mesh()
    if visual.get_num_nodes() <= MAX_VISUAL_NODES:
        return True
    return surface.is_empty() or surface.nodes_per_element != 3


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
        display_hand_models: dict[str, Path] | None = None,
        environment: str = "kitchen_sink",
        counter_height: float = DEFAULT_COUNTER_HEIGHT,
        time_budget: float | None = None,
        hand_scale: float = 1.0,
        grip_strength: float = 1.0,
    ) -> None:
        if contact_mode not in CONTACT_MODES:
            raise ValueError(f"contact_mode must be one of {CONTACT_MODES}")
        self.spec = spec
        self.roots = roots
        self.contact_mode = contact_mode
        self.min_contact_force = min_contact_force
        self.keep_self_contacts = keep_self_contacts
        self.time_step = spec.time_step
        self.scene = build_scene(spec, roots, environment, counter_height, time_budget)
        # The solver's time cap leaves room for this Python side of the step
        # (retargeting, contacts, streaming), measured as it runs.
        self._time_budget = time_budget
        self._solver_cap = time_budget * spec.time_step if time_budget else None
        self._step_seconds = 0.0
        self.environment = environment
        self.grip_strength = grip_strength
        self.workspace = workspace_for(self.scene)
        self.hands: dict[str, HandUnit] = {
            side: HandUnit(
                self.scene, roots, side, HAND_SPAWN[side], hand_variant, retarget_config,
                (display_hand_models or {}).get(side), spec.time_step, hand_scale, grip_strength,
            )
            for side in sides
        }
        self.step_count = 0
        self.sim_time = 0.0
        self._lock = threading.Lock()
        self._head_pose = np.full(7, np.nan)
        self.recorder: EpisodeRecorder | None = None
        from .scenes import render_models_for

        self._render_models = render_models_for(self.scene)
        self._collect_actors()
        scene_handles = [r.actor.get_handle() for r in self.actors if not r.hand_side]
        static_boxes = []
        for r in self.actors:
            if r.is_static and r.mesh_kind != "plane":
                box = r.actor.get_aabb_world()
                lo, hi = np.asarray(list(box.min)), np.asarray(list(box.max))
                if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi - lo < 50.0):
                    static_boxes.append((lo, hi))
        for hand in self.hands.values():
            hand.set_scene_actors(scene_handles, static_boxes)
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
            elif is_environment_actor(name):
                mesh_kind = "env"  # the client draws the environment itself
            elif deformable and _usable_visual_mesh(actor):
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
        """Latest tracking (thread-safe; the newest input wins). Hands are
        retargeted here, in the caller's thread: the server calls this as
        tracking arrives, so retargeting overlaps the physics step."""
        for side, frame in hands.items():
            if side in self.hands:
                self.hands[side].prepare(frame)
        if head_pose is not None:
            with self._lock:
                self._head_pose = np.asarray(head_pose, dtype=np.float64)

    # ------------------------------------------------------------------ step

    def step(self, with_contacts: bool = True, gather: bool = True) -> dict | None:
        """Apply the latest hand input, advance one fixed step, gather data
        (and record it when recording). Returns the gathered step data.

        Reading per-point contacts costs a few microseconds per point, so
        callers may skip it on steps that are neither recorded nor displayed
        (recorded steps always include them). With ``gather=False`` and no
        recording nothing is read back and None is returned."""
        start = time.perf_counter()
        with self._lock:
            head = self._head_pose.copy()
        for hand in self.hands.values():
            hand.apply()
        self.scene.step(self.time_step)
        link_poses = {side: hand.link_poses() for side, hand in self.hands.items()}
        for side, hand in self.hands.items():
            hand.update_unstick(link_poses[side])
        self.step_count += 1
        self._link_poses = (self.step_count, link_poses)
        self.sim_time += self.time_step
        if self.recorder is None:
            data = self.gather(head, with_contacts, full=False) if gather else None
        else:
            data = self.gather(head, True)
            self.recorder.write_step(data)
        self._adapt_budget(time.perf_counter() - start)
        return data

    def _adapt_budget(self, step_seconds: float) -> None:
        """Feedback on the solver's time cap: steer the whole step (Python
        side included: retargeting, contacts, recording) toward 92 % of the
        step period, within [MIN_SOLVER_SHARE, time_budget] of it, so the
        simulation keeps up with real time."""
        self._step_seconds += 0.1 * (step_seconds - self._step_seconds)  # average step time
        if not self._time_budget or self.step_count % 15:
            return
        dt = self.time_step
        cap = self._solver_cap + 0.5 * (0.92 * dt - self._step_seconds)
        cap = min(self._time_budget * dt, max(MIN_SOLVER_SHARE * dt, cap))
        if abs(cap - self._solver_cap) > 0.2e-3:
            params = self.scene.get_solver_params()
            params.non_linear_solver.max_elapsed_time_seconds = float(cap)
            self.scene.set_solver_params(params)
            self._solver_cap = cap

    def gather(self, head_pose: np.ndarray | None = None, with_contacts: bool = True,
               full: bool = True) -> dict:
        """The current state. ``full=False`` reads only what the viewer draws
        (no velocities or total forces; contacts without their details)."""
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
            if r.is_static or not full:
                continue  # velocities and total forces: recorded steps only
            if not r.deformable:
                lin[r.index] = _vec(a.get_linear_velocity())
                ang[r.index] = _vec(a.get_angular_velocity())
            if self._has_total_force.get(r.index):
                force[r.index] = _vec(a.get_contact_force_world())

        contacts = self._gather_contacts(full) if with_contacts else None
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
        step, cached = getattr(self, "_link_poses", (-1, {}))
        for side, hand in self.hands.items():
            frame = hand.last_input
            link_pose = cached[side] if step == self.step_count and side in cached else hand.link_poses()
            hands[side] = {
                "display_joints": hand.display.joints(link_pose),
                "tracked": hand.tracked,
                "passing_through": hand.passing_through,
                "joints": frame.joints,
                "joint_rotations": frame.rotations if frame.rotations is not None
                else np.full((hs.NUM_JOINTS, 4), np.nan),
                "joint_radii": frame.radii if frame.radii is not None
                else np.full(hs.NUM_JOINTS, np.nan),
                "target_qpos": hand.target_qpos(),
                "link_pose": link_pose,
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

    def _gather_contacts(self, full: bool = True) -> dict:
        """Contact rows; ``full=False`` reads only what the viewer draws
        (owner, other, force, position, normal) and zero-fills the rest."""
        rows: dict[str, list] = {k: [] for k in _empty_contacts()}
        threshold = self.min_contact_force
        index_of = self._index_of_handle
        side_of = self._side_of
        hand_only = self.contact_mode == "hand"
        keep_self = self.keep_self_contacts
        # Every contact is reported to both of its actors (the queried one as
        # actor_a or actor_b). Rows use the actor_a side as the owner and are
        # read from the actor_a source, except in "hand" mode, where the hand
        # links alone see every contact of interest: reading objects there
        # would walk all their contacts with the counter (thousands of points
        # for a soft body) only to discard them.
        sources = [r for r in self._contact_sources if r.hand_side] if hand_only else self._contact_sources
        for r in sources:
            handle = r.actor.get_handle().value
            for c in r.actor.get_contact_points_world():
                a = c.actor_a.value
                owner = index_of.get(a, -1)
                other = index_of.get(c.actor_b.value, -1)
                if a != handle:
                    # Reported to its actor_b: read here only if actor_a is
                    # not read (an object, in "hand" mode).
                    if not hand_only or side_of[owner] or owner < 0:
                        continue
                elif other == owner and not r.deformable:
                    continue  # rigid self pairs carry no information
                owner_side, other_side = side_of[owner], side_of[other]
                if owner_side and other_side == owner_side and not keep_self:
                    continue  # the hand touching itself
                if hand_only and not owner_side and not other_side:
                    continue
                f = c.force.tolist()
                if threshold > 0.0 and (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) < threshold**2:
                    continue
                rows["owner"].append(owner)
                rows["other"].append(other)
                rows["force"].append(f)
                rows["pos_owner"].append(c.pos_a.tolist())
                rows["normal"].append(c.normal.tolist())
                if not full:
                    continue
                rows["pos_other"].append(c.pos_b.tolist())
                rows["vel_owner"].append(c.point_velocity_a.tolist())
                rows["vel_other"].append(c.point_velocity_b.tolist())
                rows["barycentric"].append(c.parametric_coords.tolist())
                rows["distance"].append(c.distance)
                rows["int_weight"].append(c.int_weight)
                rows["element"].append(c.element_index)
                rows["sample"].append(c.sample_index)
        out = {}
        n = len(rows["owner"])
        for key, values in rows.items():
            template = _EMPTY[key]
            if len(values) != n:  # not read for display
                out[key] = np.zeros((n, *template.shape[1:]), template.dtype)
            else:
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
                        "scale": self.hands[r.hand_side].scale,
                    }
            model = self._render_models.get(r.name)
            if model is not None:
                # Prefab GLBs are Y-up exports of Z-up actor frames, like the
                # hand GLBs: rotate +90 deg about X into the actor.
                for root, route in ((self.roots.assets / "prefabs", "/prefab_assets"),
                                    (self.roots.objects, "/object_assets")):
                    try:
                        rel = model.relative_to(Path(root).resolve())
                    except ValueError:
                        continue
                    entry["render"] = {
                        "url": f"{route}/{rel.as_posix()}",
                        "rotation": [float(np.sin(np.pi / 4)), 0.0, 0.0, float(np.cos(np.pi / 4))],
                    }
                    break
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
            "hand_scale": {s: h.scale for s, h in self.hands.items()},
            "grip_strength": self.grip_strength,
            "hand_link_names": {s: h.link_names for s, h in self.hands.items()},
            "hand_dof_names": hand.kinematics.dof_names if hand else [],
            "joint_names": list(hs.JOINT_NAMES),
            "scene_description": self.spec.description,
            "environment": self.environment,
            "counter_height": self.workspace.counter_height if self.workspace else float("nan"),
            "environment_layout": self.environment_message(),
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

    def set_counter_height(self, height: float) -> None:
        """Move the floor to ``height`` below the work surface."""
        if self.workspace is not None:
            self.workspace.set_counter_height(height)

    def environment_message(self) -> dict | None:
        return self.workspace.message() if self.workspace is not None else None

    def close(self) -> None:
        self.stop_recording()
        for hand in self.hands.values():
            hand.destroy()
        forget_scene(self.scene)
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
