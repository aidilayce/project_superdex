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

"""Physically grounded objects as SuperDex prefabs.

Converts simulation assets with *measured or identified* physical parameters
into SuperDex prefabs (``.mochi_prefab`` + ``.mochi.h5`` collision shape +
textured render GLB), so hand/object forces come out right:

* **YCB objects** (the 16 YCB-Video objects from eleramp/pybullet-object-models
  plus Drake's sugar box, RobotLocomotion/models): masses measured in the YCB
  benchmark paper (Calli et al. 2015), inertia from the object's shape,
  textured scans.
* **Scalable Real2Sim** assets (Pfaff et al. 2025): mass, center of mass and
  inertia identified from robot joint torques, geometry from photometric
  reconstruction; their benchmark objects are fetched from Hugging Face.
* Any SDFormat (``.sdf``) or URDF model, or a mesh plus an inertial-parameter
  JSON (``mass``, ``center_of_mass``, ``inertia_matrix``) as Real2Sim writes.

    python -m superdex_quest_teleop.objects fetch ycb
    python -m superdex_quest_teleop.objects fetch real2sim
    python -m superdex_quest_teleop.objects convert path/to/model.sdf --material plastic
    python -m superdex_quest_teleop.objects list

Friction: no open object dataset measures friction (YCB and Real2Sim
included; simulator URDFs carry placeholder values), so friction comes from
the object's material (chosen from its name, or ``--material``), using
measured values for *dry fingertip skin* against that material from the skin
tribology literature (``MATERIALS``), because the hands' contacts are what is
recorded. SuperDex combines two actors' coefficients by their geometric
mean and the hands use 1.0, so an object's coefficient is set to
``mu_skin**2``: finger/object friction is then exactly ``mu_skin``, and
object/counter friction (counter 0.6) comes out at ``0.77 * mu_skin``, e.g.
0.44 for cardboard on stone, 0.3 for a tinned can. ``--mu`` sets the
fingertip friction directly.

Collision: SuperDex samples contact on the surfaces of both bodies, so the
collision shape should be the real surface. Primitives (boxes, cylinders)
are kept when they match the object (YCB boxes and cans); otherwise the
visual mesh is simplified to a closed mesh (the YCB mustard bottle is a box
in Drake, a bottle here); convex pieces (Real2Sim's CoACD output) are
merged when there is no usable visual mesh.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("superdex_quest_teleop")

OBJECTS_DIR = Path(os.environ.get("SUPERDEX_TELEOP_OBJECTS", Path.home() / ".superdex_quest_teleop" / "objects"))

# Dry fingertip skin friction against household materials: central values of
# the ranges reported in Derler & Gerhardt 2012 (Tribology of skin, Tribol.
# Lett. 45), Tomlinson et al. 2007 (Tribol. Int. 40) and Skedung et al. 2010
# (paper and cardboard). Skin friction varies with moisture by about +/-40 %.
MATERIALS = {
    "cardboard": 0.57,  # printed paperboard
    "paper": 0.55,
    "plastic": 0.50,  # PE/PP/ABS containers and tools
    "metal": 0.40,  # tinned or painted steel, aluminium
    "glass": 0.45,
    "ceramic": 0.45,  # glazed
    "wood": 0.50,  # varnished
    "rubber": 1.20,
    "foam": 0.90,
    "fabric": 0.45,  # felt, cotton
    "fruit": 0.55,  # waxy peel
    "leather": 0.55,
    "silicone": 1.10,
    "stone": 0.55,
    "concrete": 0.65,
    "cork": 0.60,
    "clay": 0.55,
    "carbon_fiber": 0.40,
    "default": 0.50,
}
# SceneSmith (nepfaff/scenesmith) writes a material's same-material friction
# (steel on steel 0.74, glass on glass 0.9) into its SDFs, not the material
# name: map the value back to the material(s), then use the fingertip values.
SCENESMITH_MU = {
    0.32: ("cardboard", "paper"), 0.4: ("wood", "foam", "fabric"), 1.15: ("rubber",),
    0.35: ("plastic", "ceramic"), 0.9: ("glass", "stone"), 0.6: ("leather", "cork"),
    0.74: ("metal",), 1.1: ("metal",), 0.8: ("fruit",), 1.2: ("silicone",), 0.62: ("concrete",),
    0.7: ("clay",), 0.5: ("carbon_fiber", "default"),
}
HAND_FRICTION = 1.0  # hand_rig.HAND_FRICTION: the other side of every finger contact
MATERIAL_FRICTION = MATERIALS  # backward-compatible name


def actor_friction(mu_skin: float) -> float:
    """The object's SuperDex Coulomb coefficient that makes finger contacts
    (geometric mean with the hand's HAND_FRICTION) have friction ``mu_skin``."""
    return mu_skin * mu_skin / HAND_FRICTION
_MATERIAL_KEYWORDS = (
    (("chips",), "paper"),  # a paper tube
    (("foam", "sponge"), "foam"),
    (("tennis",), "fabric"),
    (("cracker", "sugar", "gelatin", "pudding", "cereal", "cheez", "box"), "cardboard"),
    (("soup", "chef", "tuna", "potted", "meat", "can", "tin", "spam"), "metal"),
    (("mustard", "bleach", "bottle", "pitcher", "bowl", "cup", "marker", "clamp", "drill", "spatula",
      "scissors", "hammer", "plastic"), "plastic"),
    (("mug", "plate", "ceramic"), "ceramic"),
    (("banana", "apple", "lemon", "orange", "pear", "peach", "plum", "strawberry", "fruit"), "fruit"),
    (("brick", "block", "wood", "spoon"), "wood"),
    (("glass", "jar"), "glass"),
    (("toy", "rubber", "duck"), "rubber"),
    (("leather", "wallet", "shoe", "belt"), "leather"),
    (("vase", "pot", "tile"), "ceramic"),
    (("towel", "cloth", "pillow", "cushion", "napkin", "shirt"), "fabric"),
)

YCB_BASE = "https://raw.githubusercontent.com/RobotLocomotion/models/master/ycb"
# Drake's YCB SDFs; only the sugar box is not in the YCB-Video set below.
YCB_DRAKE_OBJECTS = ("004_sugar_box",)
YCB_PYBULLET_BASE = ("https://raw.githubusercontent.com/eleramp/pybullet-object-models/master/"
                     "pybullet_object_models/ycb_objects")
# The YCB-Video objects (URDF masses are the YCB paper's measured masses).
YCB_PYBULLET_OBJECTS = {
    "YcbMasterChefCan": "002_master_chef_can", "YcbCrackerBox": "003_cracker_box",
    "YcbTomatoSoupCan": "005_tomato_soup_can", "YcbMustardBottle": "006_mustard_bottle",
    "YcbGelatinBox": "009_gelatin_box", "YcbPottedMeatCan": "010_potted_meat_can",
    "YcbBanana": "011_banana", "YcbStrawberry": "012_strawberry", "YcbPear": "016_pear",
    "YcbPowerDrill": "035_power_drill", "YcbScissors": "037_scissors", "YcbHammer": "048_hammer",
    "YcbMediumClamp": "050_medium_clamp", "YcbTennisBall": "056_tennis_ball",
    "YcbFoamBrick": "061_foam_brick", "YcbChipsCan": "001_chips_can",
}
# Measured masses [kg] from the YCB paper, to check downloaded models against.
YCB_MASSES = {
    "001_chips_can": 0.205, "002_master_chef_can": 0.414, "003_cracker_box": 0.411,
    "004_sugar_box": 0.514, "005_tomato_soup_can": 0.349, "006_mustard_bottle": 0.603,
    "009_gelatin_box": 0.097, "010_potted_meat_can": 0.370, "011_banana": 0.066,
    "012_strawberry": 0.018, "016_pear": 0.049, "035_power_drill": 0.895, "037_scissors": 0.082,
    "048_hammer": 0.665, "050_medium_clamp": 0.059, "056_tennis_ball": 0.058, "061_foam_brick": 0.059,
}
REAL2SIM_REPO = "nepfaff/scalable-real2sim"
REAL2SIM_PREFIX = "scalable_real2sim_benchmark_dataset/object_data/"
SCENESMITH_REPO = "nepfaff/scenesmith-sam3d-objects"

CONTACT_SMOOTHING = 1e-3  # [m] penalty smoothing half distance
# Collision meshes are simplified to about this many triangles.
COLLISION_FACES = 2500
# Render meshes are simplified to this many triangles (Quest-friendly).
RENDER_FACES = 40000


def material_for(name: str, model_mu: float | None = None) -> str:
    """The object's material: from its name, or from a SceneSmith friction
    value (disambiguated by the name when two materials share it)."""
    lower = name.lower()
    by_name = next((m for words, m in _MATERIAL_KEYWORDS if any(w in lower for w in words)), None)
    if model_mu is not None:
        candidates = next((v for k, v in SCENESMITH_MU.items() if abs(k - model_mu) < 5e-3), None)
        if candidates:
            # 0.5 is also SceneSmith's value for an unknown material: the name wins.
            if by_name and (by_name in candidates or "default" in candidates):
                return by_name
            return candidates[0]
    return by_name or "default"


# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------


@dataclass
class ObjectModel:
    """A single rigid body, all quantities in its link frame (Z-up, meters)."""

    name: str
    mass: float
    center_of_mass: np.ndarray  # (3,)
    inertia: np.ndarray  # (3, 3) about the COM, link axes
    collision: list = field(default_factory=list)  # [(kind, params, 4x4 pose, mesh)]
    visual: list = field(default_factory=list)  # [(trimesh.Scene, 4x4 link_from_visual)]
    mu: float | None = None  # from the model, if it has one
    source: str = ""


def _pose(text: str | None) -> np.ndarray:
    """SDFormat/URDF pose ``x y z roll pitch yaw`` (fixed-axis XYZ) as 4x4."""
    T = np.eye(4)
    if not text:
        return T
    v = [float(x) for x in text.split()]
    v += [0.0] * (6 - len(v))
    r, p, y = v[3:6]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    T[:3, 3] = v[:3]
    return T


def _rx90() -> np.ndarray:
    T = np.eye(4)
    T[1:3, 1:3] = [[0.0, -1.0], [1.0, 0.0]]
    return T


def _inertia_matrix(ixx, ixy, ixz, iyy, iyz, izz) -> np.ndarray:
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], float)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _find(el, name):
    for child in el:
        if _local(child.tag) == name:
            return child
    return None


def _findall(el, name):
    return [c for c in el if _local(c.tag) == name]


def _text(el, name, default=None):
    child = _find(el, name) if el is not None else None
    return child.text.strip() if child is not None and child.text else default


def _resolve_uri(uri: str, base: Path) -> Path:
    """``package://pkg/rel``, ``model://``, ``file://`` or relative paths: look
    next to the model file and up its parents."""
    uri = uri.strip()
    for prefix in ("package://", "model://"):
        if uri.startswith(prefix):
            parts = uri[len(prefix):].split("/")
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            for root in [base, *base.parents][:4]:
                for candidate in (root / rel, root / Path(*parts)):
                    if candidate.exists():
                        return candidate
            # Drake models: package://drake_models/ycb/meshes/x -> <base>/meshes/x
            for tail in range(len(rel.parts)):
                candidate = base / Path(*rel.parts[tail:])
                if candidate.exists():
                    return candidate
            return base / rel.name
    if uri.startswith("file://"):
        uri = uri[7:]
    path = Path(uri)
    return path if path.is_absolute() else base / path


def _load_mesh_scene(path: Path, scale=(1.0, 1.0, 1.0)):
    """A mesh file as a trimesh Scene in the model's link convention: glTF is
    Y-up and gets rotated to Z-up (as Drake does); OBJ/STL/PLY are taken as is."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    T = np.diag([*[float(s) for s in scale], 1.0])
    if path.suffix.lower() in (".gltf", ".glb"):
        T = _rx90() @ T
    scene.apply_transform(T)
    return scene


def _geometry(geom_el, base: Path):
    """(kind, params, mesh_scene_or_None) of an SDFormat/URDF <geometry>."""
    for child in geom_el:
        kind = _local(child.tag)
        if kind == "box":
            size = _text(child, "size") or child.get("size")
            return "box", [float(v) for v in size.split()], None
        if kind == "sphere":
            r = _text(child, "radius") or child.get("radius")
            return "sphere", [float(r)], None
        if kind in ("cylinder", "capsule"):
            r = float(_text(child, "radius") or child.get("radius"))
            length = float(_text(child, "length") or child.get("length"))
            return kind, [r, length], None
        if kind == "ellipsoid":
            return "ellipsoid", [float(v) for v in _text(child, "radii").split()], None
        if kind == "mesh":
            uri = _text(child, "uri") or child.get("filename")
            scale = _text(child, "scale") or child.get("scale") or "1 1 1"
            s = [float(v) for v in scale.split()]
            if len(s) == 1:
                s = s * 3
            path = _resolve_uri(uri, base)
            if not path.exists():
                raise FileNotFoundError(f"mesh {uri} not found (looked for {path})")
            return "mesh", [str(path)], _load_mesh_scene(path, s)
    raise ValueError(f"unsupported geometry {[c.tag for c in geom_el]}")


def _friction(el) -> float | None:
    """<mu> (Gazebo/ODE) or drake:mu_dynamic anywhere under ``el``."""
    for node in el.iter():
        name = _local(node.tag)
        if name in ("mu_dynamic", "mu") and node.text:
            try:
                return float(node.text)
            except ValueError:
                pass
    return None


def parse_sdf(path: Path) -> ObjectModel:
    """The first link of the first model of an SDFormat file."""
    path = Path(path)
    root = ET.parse(path).getroot()
    model = next(root.iter("model"))
    links = [el for el in model if _local(el.tag) == "link"]
    if not links:
        raise ValueError(f"{path}: no <link>")
    if len(links) > 1:
        log.warning("%s: %d links, converting the first (%s)", path, len(links), links[0].get("name"))
    link = links[0]
    inertial = _find(link, "inertial")
    if inertial is None or _text(inertial, "mass") is None:
        raise ValueError(f"{path}: no <inertial><mass>: the object's mass is required")
    mass = float(_text(inertial, "mass"))
    T_in = _pose(_text(inertial, "pose"))
    I = np.zeros((3, 3))
    inertia_el = _find(inertial, "inertia")
    if inertia_el is not None:
        g = {k: float(_text(inertia_el, k, "0")) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        I = _inertia_matrix(g["ixx"], g["ixy"], g["ixz"], g["iyy"], g["iyz"], g["izz"])
    obj = ObjectModel(
        name=model.get("name") or path.stem, mass=mass, center_of_mass=T_in[:3, 3].copy(),
        inertia=T_in[:3, :3] @ I @ T_in[:3, :3].T, mu=_friction(link), source=str(path),
    )
    for col in _findall(link, "collision"):
        kind, params, scene = _geometry(_find(col, "geometry"), path.parent)
        obj.collision.append((kind, params, _pose(_text(col, "pose")), scene))
    for vis in _findall(link, "visual"):
        kind, params, scene = _geometry(_find(vis, "geometry"), path.parent)
        if scene is not None:
            obj.visual.append((scene, _pose(_text(vis, "pose"))))
    return obj


def parse_urdf(path: Path) -> ObjectModel:
    path = Path(path)
    root = ET.parse(path).getroot()
    link = next(root.iter("link"))

    def origin(el):
        o = _find(el, "origin") if el is not None else None
        if o is None:
            return np.eye(4)
        return _pose(f"{o.get('xyz', '0 0 0')} {o.get('rpy', '0 0 0')}")

    inertial = _find(link, "inertial")
    if inertial is None or _find(inertial, "mass") is None:
        raise ValueError(f"{path}: no <inertial><mass>")
    mass = float(_find(inertial, "mass").get("value"))
    T_in = origin(inertial)
    i = _find(inertial, "inertia")
    I = _inertia_matrix(*(float(i.get(k, 0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))) \
        if i is not None else np.zeros((3, 3))
    name = root.get("name") or path.stem
    if re.sub(r"\.(urdf|sdf)$", "", name.lower()) in ("model", "object", "robot", path.stem.lower()):
        name = path.parent.name  # generic robot names: use the model's folder
    obj = ObjectModel(name, mass, T_in[:3, 3].copy(),
                      T_in[:3, :3] @ I @ T_in[:3, :3].T, mu=_friction(link), source=str(path))
    for col in _findall(link, "collision"):
        kind, params, scene = _geometry(_find(col, "geometry"), path.parent)
        obj.collision.append((kind, params, origin(col), scene))
    for vis in _findall(link, "visual"):
        kind, params, scene = _geometry(_find(vis, "geometry"), path.parent)
        if scene is not None:
            obj.visual.append((scene, origin(vis)))
    return obj


def parse_mesh_with_inertia(mesh_path: Path, params_path: Path) -> ObjectModel:
    """A mesh plus a Real2Sim-style JSON {mass, center_of_mass, inertia_matrix}."""
    params = json.loads(Path(params_path).read_text())
    scene = _load_mesh_scene(Path(mesh_path))
    return ObjectModel(
        name=Path(mesh_path).parent.name if Path(mesh_path).stem in ("textured_mesh", "mesh") else Path(mesh_path).stem,
        mass=float(params["mass"]),
        center_of_mass=np.asarray(params.get("center_of_mass", [0, 0, 0]), float),
        inertia=np.asarray(params.get("inertia_matrix", np.zeros((3, 3))), float).reshape(3, 3),
        visual=[(scene, np.eye(4))],
        mu=params.get("mu_dynamic") or params.get("mu"),
        source=str(mesh_path),
    )


# ----------------------------------------------------------------------------
# Collision shape
# ----------------------------------------------------------------------------


def _primitive_mesh(kind: str, params):
    import trimesh

    if kind == "box":
        return trimesh.creation.box(extents=params)
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=params[0])
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=params[0], height=params[1], sections=48)
    if kind == "capsule":
        return trimesh.creation.capsule(height=params[1], radius=params[0], count=[24, 24])
    if kind == "ellipsoid":
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        m.apply_scale(params)
        return m
    raise ValueError(kind)


def _visual_mesh(obj: ObjectModel):
    import trimesh

    parts = []
    for scene, T in obj.visual:
        m = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)
        m.apply_transform(T)
        parts.append(m)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _simplify(mesh, faces: int):
    if len(mesh.faces) <= faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=faces)
    except Exception as exc:  # noqa: BLE001 - fast_simplification missing
        log.warning("mesh simplification unavailable (%s): pip install fast-simplification", exc)
        return mesh


def sdf_remesh(mesh, faces: int = COLLISION_FACES, pitch: float | None = None):
    """A closed, evenly tessellated copy of ``mesh`` with about ``faces``
    triangles: its signed distance sampled on a grid, then the zero level set
    (marching cubes). Keeps concave shapes (a banana, a clamp); on the YCB
    scans the result is within 0.3-0.7 mm of the original surface."""
    import trimesh
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    pitch = float(pitch or np.sqrt(2.2 * mesh.area / faces))
    lo = mesh.bounds[0] - 2 * pitch
    n = np.ceil((mesh.bounds[1] + 2 * pitch - lo) / pitch).astype(int) + 1
    axes = [lo[i] + pitch * np.arange(n[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    # Unsigned distance to a dense surface sampling (fast), sign from an
    # inside test; only cells near the surface need the exact value.
    samples, face_index = trimesh.sample.sample_surface(mesh, min(max(200000, 20 * len(mesh.faces)), 1500000))
    normals = mesh.face_normals[face_index]
    dist, nearest = cKDTree(samples).query(grid, workers=-1)
    inside = np.zeros(len(grid), bool)
    near = dist < 3 * pitch
    far = ~near
    # Near the surface: which side of the nearest surface point's normal.
    offset = grid[near] - samples[nearest[near]]
    inside[near] = np.einsum("ij,ij->i", offset, normals[nearest[near]]) < 0.0
    if far.any():
        # Far cells: the voxel fill of the mesh decides.
        filled = mesh.voxelized(pitch).fill()
        inside[far] = filled.is_filled(grid[far])
    sdf = np.where(inside, dist, -dist).reshape(n)
    verts, tris, _, _ = marching_cubes(sdf, level=0.0, spacing=(pitch, pitch, pitch))
    out = trimesh.Trimesh(verts + lo, tris[:, ::-1], process=True)
    out.merge_vertices()
    trimesh.repair.fix_normals(out)
    return out


def _closed(mesh):
    """A closed collision mesh from a (visual) mesh: an SDF remesh (accurate,
    keeps concavities), else the convex hull."""
    import trimesh

    m = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    hull = m.convex_hull
    if m.is_watertight and len(m.faces) <= COLLISION_FACES:
        trimesh.repair.fix_normals(m)
        return m, "closed mesh"
    try:
        pitch = float(np.sqrt(2.2 * m.area / COLLISION_FACES))
        if m.is_watertight:
            # Thin shells (a plate, a cup wall): the grid must resolve the wall.
            thickness = 2.0 * abs(m.volume) / m.area
            if pitch > 0.4 * thickness:
                if len(m.faces) <= 20000:
                    trimesh.repair.fix_normals(m)
                    return m, f"closed mesh (thin, {thickness * 1000:.1f} mm walls, kept as is)"
                pitch = 0.4 * thickness
        remeshed = sdf_remesh(m, pitch=pitch)
        if not remeshed.is_watertight:
            trimesh.repair.fill_holes(remeshed)
        if m.is_watertight:
            ok = abs(remeshed.volume - m.volume) < 0.25 * abs(m.volume)
        else:
            # Sanity check against garbage (thin tools fill only ~25 % of their hull).
            ok = 0.08 * hull.volume <= remeshed.volume <= 1.05 * hull.volume
        if remeshed.is_watertight and ok:
            return remeshed, "SDF-remeshed visual mesh"
        log.debug("remesh rejected (watertight %s, volume %.3g, source %.3g, hull %.3g)",
                  remeshed.is_watertight, remeshed.volume, m.volume, hull.volume)
    except Exception as exc:  # noqa: BLE001 - scikit-image/scipy missing, degenerate mesh
        log.warning("remesh failed (%s): pip install scikit-image scipy; using the convex hull", exc)
    if len(hull.faces) > COLLISION_FACES:
        hull = _simplify(hull, COLLISION_FACES).convex_hull
    return hull, "convex hull"


def collision_shape(obj: ObjectModel, prefer: str = "auto"):
    """(trimesh, collider, description) for the object's SuperDex shape.

    ``prefer``: "auto", "primitive", "visual" or "collision"."""
    import trimesh

    prims = [(k, p, T) for k, p, T, _ in obj.collision if not (k == "sphere" and p[0] < 1e-3)]
    meshes = [(s, T) for k, _, T, s in obj.collision if k == "mesh"]
    visual = _visual_mesh(obj)
    single = len(prims) == 1 and prims[0][0] in ("box", "sphere", "cylinder", "capsule", "ellipsoid")

    if single and prefer in ("auto", "primitive"):
        kind, params, T = prims[0]
        mesh = _primitive_mesh(kind, params)
        mesh.apply_transform(T)
        keep = prefer == "primitive" or visual is None
        if not keep:
            # Keep the primitive only where it fits the scanned object: every
            # side within 1.2 cm (Drake models a mustard bottle as a box 3 cm
            # shorter than the bottle), similar volume.
            hull = visual.convex_hull
            ext_ok = np.all(np.abs(np.sort(mesh.extents) - np.sort(visual.extents)) < 0.012)
            keep = ext_ok and abs(mesh.volume - hull.volume) / max(hull.volume, 1e-12) < 0.25
        if keep:
            collider = "BOX" if kind == "box" and np.allclose(T[:3, :3], np.eye(3), atol=1e-6) else "SDF"
            return mesh, collider, f"{kind} primitive"
    if visual is not None and prefer in ("auto", "visual", "primitive"):
        mesh, how = _closed(visual)
        return mesh, "SDF", f"{how}, {len(mesh.faces)} triangles"
    if meshes:
        parts = []
        for scene, T in meshes:
            m = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
            m = trimesh.Trimesh(m.vertices, m.faces)
            m.apply_transform(T)
            parts.append(m)
        merged = trimesh.util.concatenate(parts)
        mesh, how = _closed(merged)
        return mesh, "SDF", f"{len(parts)} collision meshes, {how}"
    if prims:
        parts = []
        for kind, params, T in prims:
            m = _primitive_mesh(kind, params)
            m.apply_transform(T)
            parts.append(m)
        return trimesh.util.concatenate(parts).convex_hull, "SDF", "hull of the primitives"
    raise ValueError(f"{obj.name}: no collision or visual geometry")


# ----------------------------------------------------------------------------
# Writing SuperDex prefabs
# ----------------------------------------------------------------------------


def _write_render_glb(obj: ObjectModel, path: Path) -> bool:
    """The textured visual as a Y-up GLB (the client turns prefab GLBs +90
    degrees about X into the Z-up actor frame, like the upstream prefabs)."""
    import trimesh

    if not obj.visual:
        return False
    out = trimesh.Scene()
    to_yup = np.linalg.inv(_rx90())
    for i, (scene, T) in enumerate(obj.visual):
        for node in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node]
            geom = scene.geometry[geom_name].copy()
            if len(getattr(geom, "faces", [])) > RENDER_FACES:
                simplified = _simplify(geom, RENDER_FACES)
                if simplified is not geom and hasattr(geom.visual, "uv"):
                    # Decimation drops UVs: keep the full-resolution mesh.
                    simplified = geom
                geom = simplified
            out.add_geometry(geom, node_name=f"{i}_{node}", geom_name=f"{i}_{geom_name}",
                             transform=to_yup @ T @ transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(path), file_type="glb")
    return True


def write_prefab(obj: ObjectModel, out_root: Path, material: str | None = None, mu: float | None = None,
                 collision: str = "auto", extra: dict | None = None) -> Path:
    """Convert ``obj`` into ``<out_root>/<name>/<name>.mochi_prefab`` (+ shape,
    render model and a ``physics.json`` summary). Returns the prefab path."""
    import superdex.physics as physics

    name = re.sub(r"[^A-Za-z0-9_]+", "_", obj.name).strip("_") or "object"
    out_dir = Path(out_root) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    material = material or material_for(obj.name, obj.mu)
    # Fingertip friction; model-supplied values are ignored (they are
    # simulator placeholders, e.g. 0.8 for every pybullet YCB object).
    mu_skin = float(mu if mu is not None else MATERIALS[material])
    friction = actor_friction(mu_skin)
    mesh, collider, how = collision_shape(obj, collision)

    initialized_here = not physics.is_initialized()
    if initialized_here:
        physics.initialize(num_worker_threads=1)
    scene = physics.create_scene("export")
    try:
        shape = physics.create_tri_mesh_shape(
            coordinates=np.asarray(mesh.vertices, np.float32).ravel(),
            connectivity=np.asarray(mesh.faces, np.int32).ravel(),
        )
        I = obj.inertia
        kwargs = {}
        if np.all(np.isfinite(I)) and np.trace(I) > 0:
            kwargs = dict(center_of_mass=[float(v) for v in obj.center_of_mass],
                          moment_of_inertia=[float(I[0, 0]), float(I[0, 1]), float(I[0, 2]),
                                             float(I[1, 1]), float(I[1, 2]), float(I[2, 2])])
        actor = scene.create_rigid_actor(
            name=name, shape=shape, mass=float(obj.mass),
            collider_type=getattr(physics.ColliderType, collider),
            # A 1 mm contact smoothing band (default 5 mm): objects rest within
            # a millimetre of the counter instead of sinking up to 6 mm.
            contact=physics.ContactParams(coulomb_friction_coefficient=friction,
                                          penalty_smoothing_half_distance=CONTACT_SMOOTHING),
            **kwargs,
        )
        physics.prefab.export_actor(actor, name, str(Path(out_root)))
    finally:
        physics.destroy_scene(scene)
        if initialized_here:
            physics.shutdown()

    exported = out_dir / f"{name}.mochi_scene"
    doc = json.loads(exported.read_text())
    exported.unlink()
    if _write_render_glb(obj, out_dir / "render" / f"{name}.glb"):
        for group in doc.get("actors", {}).values():
            for actor_doc in group:
                actor_doc["renderModel"] = f"./render/{name}.glb"
    prefab = out_dir / f"{name}.mochi_prefab"
    prefab.write_text(json.dumps(doc, indent=2))
    summary = {
        "name": name, "mass_kg": obj.mass, "center_of_mass_m": obj.center_of_mass.tolist(),
        "inertia_kgm2": obj.inertia.tolist(), "material": material,
        "fingertip_friction": mu_skin, "coulomb_friction": friction,
        "friction_source": "--mu" if mu is not None else "skin tribology table (MATERIALS)",
        "collision": how, "collider": collider, "extent_m": np.round(mesh.extents, 4).tolist(),
        "source": obj.source, **(extra or {}),
    }
    (out_dir / "physics.json").write_text(json.dumps(summary, indent=2))
    log.info("%s: %.3f kg, fingertip mu %.2f (%s), collision %s", name, obj.mass, mu_skin, material, how)
    return prefab


def convert(path: Path, out_root: Path, **kwargs) -> Path:
    path = Path(path)
    if path.suffix == ".sdf":
        obj = parse_sdf(path)
    elif path.suffix == ".urdf":
        obj = parse_urdf(path)
    else:
        raise ValueError(f"{path}: expected an .sdf or .urdf file")
    return write_prefab(obj, out_root, **kwargs)


# ----------------------------------------------------------------------------
# Fetching object sets
# ----------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


def fetch_ycb(out_root: Path = OBJECTS_DIR / "ycb", cache: Path | None = None) -> list[Path]:
    """17 YCB objects (CC BY 4.0) as SuperDex prefabs: the YCB-Video set from
    pybullet-object-models (textured scans, measured masses) and Drake's
    sugar box. Masses are checked against the YCB paper."""
    cache = cache or out_root.parent / "_downloads" / "ycb"
    prefabs = []
    for folder, name in YCB_PYBULLET_OBJECTS.items():
        base = f"{YCB_PYBULLET_BASE}/{folder}"
        local = cache / name
        urdf = _download(f"{base}/model.urdf", local / "model.urdf")
        text = urdf.read_text()
        for ref in sorted(set(re.findall(r'filename="([^"]+)"', text))):
            mesh = _download(f"{base}/{ref}", local / ref)
            if mesh.suffix == ".obj":
                for mtl in re.findall(r"^mtllib\s+(\S+)", mesh.read_text(errors="ignore"), re.M):
                    mtl_path = _download(f"{base}/{mtl}", local / mtl)
                    for tex in re.findall(r"map_Kd\s+(\S+)", mtl_path.read_text(errors="ignore")):
                        _download(f"{base}/{tex}", local / tex)
        obj = parse_urdf(urdf)
        obj.name = name
        prefabs.append(_ycb_prefab(obj, out_root, f"eleramp/pybullet-object-models {folder}"))
    for name in YCB_DRAKE_OBJECTS:
        sdf = _download(f"{YCB_BASE}/{name}.sdf", cache / "drake" / f"{name}.sdf")
        for ext in ("gltf", "bin", "png"):
            _download(f"{YCB_BASE}/meshes/{name}_textured.{ext}", cache / "drake" / "meshes" / f"{name}_textured.{ext}")
        prefabs.append(_ycb_prefab(parse_sdf(sdf), out_root, "RobotLocomotion/models ycb"))
    _write_license(out_root, "YCB object and model set (Calli et al. 2015, http://ycbbenchmarks.org), "
                   "CC BY 4.0. Models from https://github.com/eleramp/pybullet-object-models (LGPL-2.1) "
                   "and https://github.com/RobotLocomotion/models. Converted to SuperDex prefabs.")
    return prefabs


def _ycb_prefab(obj: ObjectModel, out_root: Path, origin: str) -> Path:
    paper = YCB_MASSES.get(obj.name)
    if paper is not None and abs(obj.mass - paper) > 0.02 * paper + 0.001:
        log.warning("%s: model mass %.3f kg differs from the YCB paper's %.3f kg: using the paper's",
                    obj.name, obj.mass, paper)
        obj.mass = paper
    return write_prefab(obj, out_root, extra={
        "license": "CC BY 4.0 (YCB object and model set)", "origin": origin,
        "mass_source": "YCB paper (measured)" if paper is not None else "model file"})


def fetch_hf(repo: str, prefix: str, out_root: Path, cache: Path, match: str | None = None,
             limit: int | None = None, license_note: str = "") -> list[Path]:
    """Convert the models of a Hugging Face dataset, downloading only what they
    reference: every ``.sdf``/``.urdf`` under ``prefix`` (filtered by
    ``match``/``limit``), the meshes they name, and the materials and
    textures of those meshes. Datasets without SDF/URDF files are searched
    for meshes with Real2Sim ``*inertial_params.json`` files instead."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix)]
    if not files:
        raise SystemExit(f"no files under {prefix!r} in {repo}")
    available = set(files)

    def get(rel: str) -> Path | None:
        if rel not in available:
            return None
        return Path(hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(cache)))

    def get_with_deps(rel: str) -> None:
        path = get(rel)
        if path is None:
            return
        folder = Path(rel).parent
        text = path.read_text(errors="ignore") if path.suffix in (".sdf", ".urdf", ".obj", ".mtl", ".gltf") else ""
        refs = []
        if path.suffix in (".sdf", ".urdf"):
            refs = re.findall(r"<uri>\s*([^<\s]+)\s*</uri>", text) + re.findall(r'filename="([^"]+)"', text)
            refs = [re.sub(r"^(package|model)://[^/]+/", "", r) for r in refs]
        elif path.suffix == ".obj":
            refs = re.findall(r"^mtllib\s+(\S+)", text, re.M)
        elif path.suffix == ".mtl":
            refs = re.findall(r"map_\w+\s+(?:-\S+\s+\S+\s+)*(\S+)", text)
        elif path.suffix == ".gltf":
            refs = re.findall(r'"uri"\s*:\s*"([^"]+)"', text)
        for ref in refs:
            dep = (folder / ref).as_posix()
            if dep in available and not (cache / dep).exists():
                get_with_deps(dep)

    models = sorted(f for f in files if f.endswith((".sdf", ".urdf")) and "_scaled" not in f)
    if match:
        models = [f for f in models if match.lower() in f.lower()]
    if limit:
        models = models[:limit]
    if models:
        log.info("%s: converting %d of the %d SDF/URDF models", repo, len(models),
                 sum(f.endswith((".sdf", ".urdf")) for f in files))
        for rel in models:
            get_with_deps(rel)
        prefabs = []
        for rel in models:
            try:
                prefabs.append(convert(cache / rel, out_root, extra={"license": license_note, "origin": repo}))
            except Exception as exc:  # noqa: BLE001 - report and continue
                log.warning("skipping %s: %s", rel, exc)
        _write_license(out_root, license_note)
        return prefabs
    # Real2Sim-style folders: textured mesh + inertial parameters.
    params = sorted(f for f in files if f.endswith("inertial_params.json"))
    if match:
        params = [f for f in params if match.lower() in f.lower()]
    if limit:
        params = params[:limit]
    for rel in params:
        get(rel)
        folder = Path(rel).parent.as_posix()
        meshes = [f for f in files if f.startswith(folder) and f.endswith("textured_mesh.obj")] or \
                 [f for f in files if f.startswith(folder) and f.endswith(".obj")][:1]
        for m in meshes[:1]:
            get_with_deps(m)
    if not params:
        raise SystemExit(f"{repo}: no SDF/URDF models or inertial-parameter files under {prefix!r}; "
                         f"file types: {sorted({Path(f).suffix for f in files})}")
    return convert_tree(cache / prefix, out_root, license_note=license_note)


def fetch_real2sim(out_root: Path = OBJECTS_DIR / "real2sim", cache: Path | None = None,
                   match: str | None = None, limit: int | None = None) -> list[Path]:
    """The Scalable Real2Sim benchmark objects (identified mass and inertia)."""
    return fetch_hf(REAL2SIM_REPO, REAL2SIM_PREFIX, out_root,
                    cache or out_root.parent / "_downloads" / "real2sim", match, limit,
                    f"Scalable Real2Sim benchmark dataset (Pfaff et al. 2025), huggingface.co/datasets/{REAL2SIM_REPO}")


def fetch_scenesmith(out_root: Path = OBJECTS_DIR / "scenesmith", cache: Path | None = None,
                     match: str | None = None, limit: int | None = 30) -> list[Path]:
    """SceneSmith's generated objects (SAM-3D geometry; VLM-estimated mass and
    material, so plausible rather than measured)."""
    return fetch_hf(SCENESMITH_REPO, "", out_root, cache or out_root.parent / "_downloads" / "scenesmith",
                    match, limit, f"SceneSmith SAM-3D objects (Pfaff et al. 2026), huggingface.co/datasets/{SCENESMITH_REPO}")


def convert_tree(root: Path, out_root: Path, license_note: str = "") -> list[Path]:
    """Convert every model under ``root``: SDF/URDF files, else meshes with a
    ``*inertial_params.json`` next to them."""
    root = Path(root)
    prefabs, done = [], set()
    for model in sorted(list(root.rglob("*.sdf")) + list(root.rglob("*.urdf"))):
        try:
            prefabs.append(convert(model, out_root, extra={"license": license_note}))
            done.add(model.parent)
        except Exception as exc:  # noqa: BLE001 - report and continue
            log.warning("skipping %s: %s", model, exc)
    for params in sorted(root.rglob("*inertial_params.json")):
        folder = params.parent
        if folder in done or any(d in folder.parents for d in done):
            continue
        meshes = sorted(folder.rglob("textured_mesh.obj")) or sorted(folder.rglob("*.obj"))
        if not meshes:
            continue
        try:
            obj = parse_mesh_with_inertia(meshes[0], params)
            obj.name = folder.name
            prefabs.append(write_prefab(obj, out_root, extra={"license": license_note}))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", folder, exc)
    if license_note:
        _write_license(out_root, license_note)
    return prefabs


def _write_license(out_root: Path, text: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "SOURCE.txt").write_text(text + "\n")


def installed_objects(root: Path = OBJECTS_DIR) -> dict[str, list[Path]]:
    """Converted prefabs by object set."""
    out = {}
    if root.exists():
        for set_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            prefabs = sorted(set_dir.glob("*/*.mochi_prefab"))
            if prefabs:
                out[set_dir.name] = prefabs
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m superdex_quest_teleop.objects", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OBJECTS_DIR, help="object library (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and convert an object set")
    f.add_argument("set", choices=("ycb", "real2sim", "scenesmith", "all"))
    f.add_argument("--match", default=None, help="only models whose path contains this text")
    f.add_argument("--limit", type=int, default=None, help="at most this many models (scenesmith: 30)")
    c = sub.add_parser("convert", help="convert SDF/URDF files or folders")
    c.add_argument("paths", nargs="+", type=Path)
    c.add_argument("--set", default="custom", help="object set (subfolder) to add them to")
    c.add_argument("--material", choices=sorted(MATERIALS), default=None,
                   help="sets the fingertip friction (default: guessed from the name)")
    c.add_argument("--mu", type=float, default=None, help="fingertip (skin) friction, overrides the material")
    c.add_argument("--collision", choices=("auto", "primitive", "visual", "collision"), default="auto")
    sub.add_parser("list", help="show the converted objects and their physics")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "fetch":
        if args.set in ("ycb", "all"):
            fetch_ycb(args.out / "ycb")
        if args.set in ("real2sim", "all"):
            fetch_real2sim(args.out / "real2sim", match=args.match, limit=args.limit)
        if args.set in ("scenesmith", "all"):
            fetch_scenesmith(args.out / "scenesmith", match=args.match, limit=args.limit or 30)
    elif args.cmd == "convert":
        for p in args.paths:
            if p.is_dir():
                convert_tree(p, args.out / args.set)
            else:
                convert(p, args.out / args.set, material=args.material, mu=args.mu, collision=args.collision)
    if args.cmd in ("fetch", "convert", "list"):
        for set_name, prefabs in installed_objects(args.out).items():
            print(f"{set_name}:")
            for prefab in prefabs:
                info = json.loads((prefab.parent / "physics.json").read_text())
                mu = info.get("fingertip_friction", info["coulomb_friction"])
                print(f"  {info['name']:24s} {info['mass_kg'] * 1000:7.1f} g  fingertip mu {mu:.2f} "
                      f"({info['material']})  {info['collision']}")
        print("Each object is a scene (obj_<set>_<name>); the sets are scenes too (objects_<set>).")


if __name__ == "__main__":
    main()
