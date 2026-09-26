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

"""Teleoperation scenes built from the SuperDex asset library.

Every scene uses the SuperDex Physics convention (Y-up, meters, gravity -Y)
with a static tabletop plane at y = 0 centered on the origin. The VR client
places this origin on a virtual table in front of the operator.

Prefabs under ``assets/prefabs`` are authored Z-up; like the visionOS
prefab gallery they are rotated -90 degrees about X, then automatically
centered on the table (measured from their actors' bounds) so any new
prefab dropped into the asset library shows up as a scene without extra
code (see ``discover_prefab_scenes``).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import superdex.physics as physics

from .workspace import DEFAULT_COUNTER_HEIGHT, Workspace

TABLE_NAME = "table"


def default_repo_root() -> Path:
    """The SuperDex checkout this package lives in (override: SUPERDEX_REPO)."""
    env = os.environ.get("SUPERDEX_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass
class AssetRoots:
    """Where scene assets live in a SuperDex checkout."""

    repo: Path = field(default_factory=default_repo_root)

    @property
    def assets(self) -> Path:
        """Top-level robotics assets (bots, prefabs)."""
        return self.repo / "assets"

    @property
    def physics_assets(self) -> Path:
        """superdex_physics/assets (meshes used by the physics examples)."""
        return self.repo / "superdex_physics" / "assets"

    @property
    def objects(self) -> Path:
        """Converted object library (python -m superdex_quest_teleop.objects)."""
        from .objects import OBJECTS_DIR

        return OBJECTS_DIR


@dataclass
class SceneSpec:
    """A teleoperation scene the operator can load."""

    id: str
    name: str
    category: str  # "rigid", "deformable" or "mixed"
    description: str
    build: Callable[[physics.Scene, AssetRoots], None]
    time_step: float = 1.0 / 60.0
    # Solver tweaks for deformables (line search as in the rope/cloth examples).
    robust_solver: bool = False
    # Newton iterations per step (0: engine default). Interactive shells use 2
    # like example_tshirt_on_plane.py.
    newton_iters: int = 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def add_table(scene: physics.Scene) -> physics.Actor:
    """Static tabletop: an infinite plane through the origin, normal +Y."""
    return scene.create_rigid_actor(
        name=TABLE_NAME,
        shape=physics.create_plane_shape(normal=[0, 1, 0], distance=0.0),
        is_static=True,
    )


def _actors(scene: physics.Scene) -> list[physics.Actor]:
    out: list[physics.Actor] = []
    scene.for_each_actor(lambda a: out.append(a))
    return out


def _world_bounds(actors: list[physics.Actor]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for actor in actors:
        try:
            box = actor.get_aabb_world()
        except Exception:  # noqa: BLE001 - actors without collision (e.g. joint-only links)
            continue
        lo = np.minimum(lo, list(box.min))
        hi = np.maximum(hi, list(box.max))
    return lo, hi


_PREFAB_ROTATION = physics.Quaternion.rotation_x(-math.pi / 2)

# Soft rubber and silicone on skin: fingertip friction about 1.0 (objects.py
# MATERIALS); with the hands' 1.0 the pair coefficient is sqrt(1.0 * 1.0).
SILICONE_FRICTION = 1.0

# Textured render models of prefab actors, per scene handle:
# {actor name: GLB path}. Filled by add_prefab, read by the session.
_render_models: dict[int, dict[str, Path]] = {}


def prefab_render_models(prefab_path: Path, prefix: str) -> dict[str, Path]:
    """Actor name -> render GLB for a prefab loaded under ``prefix`` (the
    PrefabParams name), following nested prefabs ("prefab/<nested>/<actor>")."""
    out: dict[str, Path] = {}
    try:
        doc = json.loads(Path(prefab_path).read_text())
    except (OSError, ValueError):
        return out
    base = Path(prefab_path).parent
    for actors in (doc.get("actors") or {}).values():
        for actor in actors if isinstance(actors, list) else []:
            model = actor.get("renderModel")
            name = actor.get("name")
            if model and name and (base / model).exists():
                out[f"{prefix}/{name}"] = (base / model).resolve()
    for nested in doc.get("prefabs") or []:
        path = nested.get("path")
        if path and nested.get("name"):
            out.update(prefab_render_models(base / path, f"{prefix}/{nested['name']}"))
    return out


def render_models_for(scene: physics.Scene) -> dict[str, Path]:
    return _render_models.get(scene.get_handle().value, {})
_placement_cache: dict[str, np.ndarray] = {}


def prefab_placement(prefab_path: Path, clearance: float = 0.002) -> np.ndarray:
    """Translation that centers a Z-up prefab on the table (after the -90 deg
    X rotation): XZ bounds centered on the origin, lowest point
    ``clearance`` above the table. Measured once in a scratch scene."""
    key = f"{prefab_path}:{clearance}"
    cached = _placement_cache.get(key)
    if cached is not None:
        return cached
    scratch = physics.create_scene("placement")
    try:
        physics.prefab.add_to_scene(
            prefab_path=str(prefab_path),
            root_path=str(prefab_path.parent),
            scene=scratch,
            params=physics.prefab.PrefabParams(name="p", rotation=_PREFAB_ROTATION),
        )
        lo, hi = _world_bounds(_actors(scratch))
    finally:
        physics.destroy_scene(scratch)
    center = 0.5 * (lo + hi)
    offset = np.array([-center[0], clearance - lo[1], -center[2]])
    _placement_cache[key] = offset
    return offset


def add_prefab(
    scene: physics.Scene,
    prefab_path: Path,
    name: str = "prefab",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    clearance: float = 0.002,
    friction: dict[str, float] | None = None,
) -> None:
    """Add a prefab centered on the table, then shifted by ``offset``.

    ``friction`` overrides the Coulomb coefficient of the prefab's actors
    named by it (actor name without the prefab prefix)."""
    placement = prefab_placement(prefab_path, clearance) + np.asarray(offset)
    _render_models.setdefault(scene.get_handle().value, {}).update(
        prefab_render_models(prefab_path, name)
    )
    physics.prefab.add_to_scene(
        prefab_path=str(prefab_path),
        root_path=str(prefab_path.parent),
        scene=scene,
        params=physics.prefab.PrefabParams(
            name=name,
            rotation=_PREFAB_ROTATION,
            translation=[float(v) for v in placement],
        ),
    )
    if friction:
        def override(actor: physics.Actor) -> None:
            local = actor.get_name().removeprefix(f"{name}/")
            if local in friction:
                params = actor.get_contact_params()
                params.coulomb_friction_coefficient = float(friction[local])
                actor.set_contact_params(params)

        scene.for_each_actor(override)


def _use_robust_solver(scene: physics.Scene) -> None:
    params = scene.get_solver_params()
    params.non_linear_solver.line_search_type = physics.LineSearchType.WOLFE_STRONG
    params.experimental_eval.implicit_normal_force_for_dissipation = True
    scene.set_solver_params(params)


def _tune_solver(scene: physics.Scene, spec: SceneSpec, time_budget: float | None) -> None:
    params = scene.get_solver_params()
    # An inner linear solve that stops short is corrected by the next Newton
    # iteration or step; at teleop rates its warnings only flood the terminal.
    params.linear_solver.verbosity = physics.VerbosityLevel.ERROR
    if spec.newton_iters > 0:
        params.non_linear_solver.max_iter = spec.newton_iters
    if time_budget is not None and time_budget > 0.0:
        params.non_linear_solver.max_elapsed_time_seconds = float(time_budget * spec.time_step)
    scene.set_solver_params(params)


def _soft_material(youngs: float, poisson: float, density: float):
    return physics.SoftMaterialParams(
        type=physics.SoftMaterialType.NEO_HOOKEAN,
        neo_hookean=physics.NeoHookeanMaterialParams(
            youngs_modulus=youngs, poisson_ratio=poisson
        ),
        density=density,
    )


def add_soft_mesh(
    scene: physics.Scene,
    name: str,
    shape_path: Path,
    size: float,
    position: tuple[float, float, float],
    youngs: float,
    poisson: float = 0.35,
    density: float = 400.0,
    friction: float = 0.6,
) -> physics.Actor:
    """A tet-mesh soft body scaled so its largest side is ``size`` meters,
    resting with its lowest point at ``position``'s height."""
    probe = physics.load_shape_from_file(file_path=str(shape_path))
    box = physics.get_shape_aabb(probe)
    lo, hi = np.asarray(list(box.min)), np.asarray(list(box.max))
    scale = size / float(np.max(hi - lo))
    shape = physics.load_shape_from_file(
        file_path=str(shape_path), bake_scale=[scale] * 3
    )
    center = 0.5 * (lo + hi) * scale
    translation = [
        position[0] - center[0],
        position[1] - lo[1] * scale,
        position[2] - center[2],
    ]
    return scene.create_soft_actor(
        name=name,
        shape=shape,
        material=_soft_material(youngs, poisson, density),
        contact=physics.ContactParams(coulomb_friction_coefficient=friction),
        world_from_local=physics.TransformRT(translation=translation),
    )


def add_soft_box(
    scene: physics.Scene,
    name: str,
    shape_path: Path,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    youngs: float,
    poisson: float = 0.25,
    density: float = 80.0,
    friction: float = 0.9,
) -> physics.Actor:
    """A soft box: a unit tet cube baked to ``size`` [m], resting on the
    table at ``position`` (x, z centered, lowest point at position's y)."""
    probe = physics.load_shape_from_file(file_path=str(shape_path))
    box = physics.get_shape_aabb(probe)
    lo, hi = np.asarray(list(box.min)), np.asarray(list(box.max))
    scale = np.asarray(size) / (hi - lo)
    shape = physics.load_shape_from_file(
        file_path=str(shape_path), bake_scale=[float(v) for v in scale]
    )
    center = 0.5 * (lo + hi) * scale
    return scene.create_soft_actor(
        name=name,
        shape=shape,
        material=_soft_material(youngs, poisson, density),
        contact=physics.ContactParams(coulomb_friction_coefficient=friction),
        world_from_local=physics.TransformRT(
            translation=[position[0] - center[0], position[1] - lo[1] * scale[1], position[2] - center[2]]
        ),
    )


def add_rigid_mesh(
    scene: physics.Scene,
    name: str,
    shape_path: Path,
    size: float,
    position: tuple[float, float, float],
    density: float = 500.0,
    collider: physics.ColliderType = physics.ColliderType.AUTO,
    friction: float = 0.6,
) -> physics.Actor:
    probe = physics.load_shape_from_file(file_path=str(shape_path))
    box = physics.get_shape_aabb(probe)
    lo, hi = np.asarray(list(box.min)), np.asarray(list(box.max))
    scale = size / float(np.max(hi - lo))
    shape = physics.load_shape_from_file(
        file_path=str(shape_path), bake_scale=[scale] * 3
    )
    center = 0.5 * (lo + hi) * scale
    return scene.create_rigid_actor(
        name=name,
        shape=shape,
        density=density,
        collider_type=collider,
        contact=physics.ContactParams(coulomb_friction_coefficient=friction),
        world_from_local=physics.TransformRT(
            translation=[
                position[0] - center[0],
                position[1] - lo[1] * scale,
                position[2] - center[2],
            ]
        ),
    )


def add_free_rope(
    scene: physics.Scene,
    name: str = "rope",
    length: float = 0.6,
    radius: float = 0.006,
    num_elements: int = 64,
    height: float = 0.03,
) -> physics.Actor:
    """The visionOS example's free rope: a tubular rod on the table."""
    density, youngs, shear = 500.0, 5e6, 2e6
    t = np.linspace(0.0, 1.0, num_elements + 1)
    nodes = np.stack(
        [
            -0.5 * length + t * length,
            np.full_like(t, height),
            0.05 * np.sin(2.0 * np.pi * t),
        ],
        axis=1,
    ).astype(np.float32)
    axes = np.tile([0.0, 1.0, 0.0], (num_elements, 1)).astype(np.float32)
    model = physics.experimental.generate_tubular_rod_model_data(
        nodes=nodes,
        element_frame_axes=axes,
        radius=radius,
        num_cross_section_segments=8,
        is_closed_loop=False,
    )
    area = math.pi * radius**2
    polar = 0.5 * math.pi * radius**4
    second = 0.5 * polar
    material = physics.experimental.RodMaterialParams(
        linear_density=density * area,
        linear_rotational_inertia=density * polar,
        axial_stiffness=youngs * area,
        torsional_stiffness=shear * polar,
        flexural_stiffness=[youngs * second, youngs * second],
    )
    params = dict(
        name=name,
        shape=physics.create_model_shape(model),
        material=material,
        contact=physics.ContactParams(penalty_coefficient=1e8, coulomb_friction_coefficient=0.5),
    )
    # Contact on the tube surface: `use_contact_skin` on current SuperDex,
    # `use_visual_mesh_contact` on the 1.0.0 wheels.
    try:
        rod = physics.experimental.RodActorParams(**params, use_contact_skin=True)
    except TypeError:
        rod = physics.experimental.RodActorParams(**params, use_visual_mesh_contact=True)
    return physics.experimental.create_rod_actor(scene, rod)


def add_cloth(
    scene: physics.Scene,
    roots: AssetRoots,
    name: str = "tshirt",
    size: float = 0.45,
    height: float = 0.06,
    radius: float = 0.003,
    self_contact: bool = True,
) -> physics.Actor:
    """The t-shirt shell from example_tshirt_on_plane.py, scaled to ``size``.

    ``radius`` is the point-cloud contact radius; ``self_contact`` lets the
    garment collide with itself (folds), the most expensive part."""
    path = roots.physics_assets / "garments" / "tshirt_visual_subdiv_2.mochi.h5"
    probe = physics.load_shape_from_file(file_path=str(path))
    box = physics.get_shape_aabb(probe)
    lo, hi = np.asarray(list(box.min)), np.asarray(list(box.max))
    scale = size / float(np.max(hi - lo))
    shape = physics.load_shape_from_file(file_path=str(path), bake_scale=[scale] * 3)
    center = 0.5 * (lo + hi) * scale
    material = physics.experimental.shell_material_params_from3d_isotropic(
        1e5, 0.3, 300.0, 0.002
    )
    # Lay the garment flat: its authored front faces +Z, rotate to face +Y.
    params = physics.experimental.ShellActorParams(
        name=name,
        shape=shape,
        material=material,
        world_from_local=physics.TransformRT(
            rotation=physics.Quaternion.rotation_x(-math.pi / 2),
            translation=[-center[0], height - lo[2] * scale, center[1]],
        ),
    )
    params.point_cloud_collider.radius = radius
    params.point_cloud_collider.self_contact = self_contact
    return physics.experimental.create_shell_actor(scene, params)


def add_towel(
    scene: physics.Scene,
    name: str = "towel",
    size: tuple[float, float] = (0.3, 0.3),
    cells: int = 24,
    height: float = 0.01,
    center: tuple[float, float] = (0.0, 0.0),
) -> physics.Actor:
    """A flat cotton dish towel: a ``cells`` x ``cells`` triangle-grid shell
    lying on the counter (light enough for real time on any machine)."""
    n = cells + 1
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n), indexing="ij")
    nodes = np.stack([u.ravel() * size[0], np.zeros(n * n), v.ravel() * size[1]], axis=1)
    tris = []
    for i in range(cells):
        for j in range(cells):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            # Alternate the diagonal so the fabric has no preferred shear.
            tris += [[a, d, b], [b, d, c]] if (i + j) % 2 == 0 else [[a, d, c], [a, c, b]]
    shape = physics.create_tri_mesh_shape(
        coordinates=nodes.astype(np.float32).ravel(),
        connectivity=np.asarray(tris, np.int32).ravel(),
    )
    # Cotton terry: ~1 mm effective thickness, ~300 g/m^2.
    material = physics.experimental.shell_material_params_from3d_isotropic(2e5, 0.3, 300.0, 0.001)
    params = physics.experimental.ShellActorParams(
        name=name,
        shape=shape,
        material=material,
        contact=physics.ContactParams(coulomb_friction_coefficient=0.8),
        world_from_local=physics.TransformRT(translation=[center[0], height, center[1]]),
    )
    params.point_cloud_collider.radius = 0.004
    params.point_cloud_collider.self_contact = True
    return physics.experimental.create_shell_actor(scene, params)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

# Prefab scenes with curated names; any other prefab in assets/prefabs is
# still discovered automatically.
_PREFAB_SCENES: tuple[tuple[str, str, str, str, dict], ...] = (
    ("box_and_blocks", "Box and Blocks", "box_and_blocks/box_and_blocks.mochi_prefab",
     "Transfer colored blocks across the partition (rigid).", {}),
    # Silicone: 1.0 (pair 1.0 with the hand; the asset's 0.25 gives 0.5,
    # glass-like, and the 545 g lamp slides out of any grasp).
    ("duck_lamp", "Duck Lamp (soft)", "duck_lamp/duck_lamp_recumbent.mochi_prefab",
     "Squeeze and lift a soft neo-Hookean duck lamp (deformable).",
     {"friction": {"DuckLamp": SILICONE_FRICTION}}),
    ("sphere", "Sphere", "sphere/sphere.mochi_prefab",
     "Roll, pinch and pick a 3 cm sphere (rigid).", {}),
    ("shape_box", "Shape Sorter Box", "shape_box/shape_box.mochi_prefab",
     "Sort 11 shapes and open the lid (rigid).", {}),
    ("nine_hole_peg_test", "Nine Hole Peg Test", "nine_hole_peg_test/nine_hole_peg_test.mochi_prefab",
     "Clinical dexterity test: pick and insert thin pegs (rigid).", {}),
    ("functional_dexterity_test", "Functional Dexterity Test",
     "functional_dexterity_test/functional_dexterity_test.mochi_prefab",
     "Flip and reinsert 16 pegs (rigid).", {}),
    ("fdt_peg", "Single FDT Peg", "functional_dexterity_test/fdt_peg.mochi_prefab",
     "A single cylindrical peg (rigid).", {}),
    ("paper_cup", "Paper Cup", "paper_cups/paper_cup.mochi_prefab",
     "Grasp a thin-walled cup (rigid).", {}),
    ("paper_cup_pyramid", "Paper Cup Pyramid", "paper_cups/paper_cup_pyramid.mochi_prefab",
     "Knock down or unstack a 10-cup pyramid (rigid).", {}),
    ("chain", "Hanging Chain", "chain/chain.mochi_prefab",
     "Swing and grab a 10-link chain hanging from a fixed link (rigid).",
     {"clearance": 0.06}),
    ("block_red", "Single Block", "box_and_blocks/block_red.mochi_prefab",
     "One 2.5 cm cube block (rigid).", {}),
)


def _prefab_builder(relative: str, **kwargs) -> Callable[[physics.Scene, AssetRoots], None]:
    def build(scene: physics.Scene, roots: AssetRoots) -> None:
        add_prefab(scene, roots.assets / "prefabs" / relative, **kwargs)

    return build


def _build_rigid_cube(scene: physics.Scene, roots: AssetRoots) -> None:
    add_rigid_mesh(
        scene, "cube", roots.assets / "cube" / "cube_fine_mesh.mochi.h5",
        size=0.057, position=(0.0, 0.0, 0.0), density=400.0,
        collider=physics.ColliderType.BOX,
    )


def _build_soft_cube(scene: physics.Scene, roots: AssetRoots) -> None:
    add_soft_mesh(
        scene, "soft_cube", roots.physics_assets / "cube" / "cube_fine_mesh.mochi.json",
        size=0.06, position=(0.0, 0.0, 0.0), youngs=3e4, density=100.0,
    )


def _build_soft_duck(scene: physics.Scene, roots: AssetRoots) -> None:
    add_soft_mesh(
        scene, "soft_duck", roots.physics_assets / "duck" / "duck_730.mochi.h5",
        size=0.12, position=(0.0, 0.0, 0.0), youngs=2e4, density=200.0, friction=SILICONE_FRICTION,
    )


def _build_rope(scene: physics.Scene, roots: AssetRoots) -> None:
    add_free_rope(scene)


def _build_cloth(scene: physics.Scene, roots: AssetRoots) -> None:
    add_cloth(scene, roots)


def _build_towel(scene: physics.Scene, roots: AssetRoots) -> None:
    add_towel(scene)


def _build_kitchen_sponge(scene: physics.Scene, roots: AssetRoots) -> None:
    """A kitchen sponge (soft foam block) and a paper cup on the countertop,
    in front of the sink of the kitchen_sink environment."""
    add_soft_box(
        scene, "sponge", roots.physics_assets / "cube" / "cube_fine_mesh.mochi.json",
        size=(0.095, 0.032, 0.065), position=(0.05, 0.0, 0.0), youngs=1.2e4, density=80.0,
    )
    add_prefab(scene, roots.assets / "prefabs" / "paper_cups" / "paper_cup.mochi_prefab",
               "cup", (-0.17, 0.0, -0.08))


def _build_medley(scene: physics.Scene, roots: AssetRoots) -> None:
    """A tabletop mix of rigid and deformable objects."""
    prefabs = roots.assets / "prefabs"
    add_prefab(scene, prefabs / "sphere" / "sphere.mochi_prefab", "sphere", (-0.12, 0.0, 0.0))
    add_prefab(scene, prefabs / "paper_cups" / "paper_cup.mochi_prefab", "cup", (0.12, 0.0, -0.05))
    add_prefab(scene, prefabs / "box_and_blocks" / "block_red.mochi_prefab", "block", (0.0, 0.0, 0.08))
    add_prefab(scene, prefabs / "functional_dexterity_test" / "fdt_peg.mochi_prefab", "peg", (-0.05, 0.0, -0.08))
    add_rigid_mesh(
        scene, "cube", roots.assets / "cube" / "cube_fine_mesh.mochi.h5",
        size=0.057, position=(0.1, 0.0, 0.08), collider=physics.ColliderType.BOX,
    )
    add_soft_mesh(
        scene, "soft_duck", roots.physics_assets / "duck" / "duck_730.mochi.h5",
        size=0.1, position=(0.0, 0.0, -0.02), youngs=2e4, density=200.0,
    )


_BUILTIN: tuple[SceneSpec, ...] = (
    SceneSpec("kitchen_sponge", "Kitchen Sponge", "deformable",
              "Squeeze and wipe with a soft kitchen sponge next to the sink.",
              _build_kitchen_sponge, robust_solver=True),
    SceneSpec("cube", "Cube (rigid)", "rigid",
              "A 5.7 cm puzzle-cube sized rigid block.", _build_rigid_cube),
    SceneSpec("soft_cube", "Soft Cube", "deformable",
              "A squishy 6 cm foam cube (neo-Hookean tet mesh).", _build_soft_cube,
              robust_solver=True),
    SceneSpec("soft_duck", "Soft Duck", "deformable",
              "The soft duck from example_soft_duck.py at 12 cm.", _build_soft_duck,
              robust_solver=True),
    SceneSpec("free_rope", "Free Rope", "deformable",
              "The visionOS free rope: lift, drag and coil a 60 cm rod.", _build_rope,
              robust_solver=True),
    # One Newton iteration per step (example_tshirt_on_plane.py uses 2) keeps
    # the 3.6k-node garment with self contact near real time.
    SceneSpec("cloth", "T-shirt Cloth", "deformable",
              "A 45 cm t-shirt shell with self contact (the heaviest scene).",
              _build_cloth, time_step=1.0 / 30.0, robust_solver=True, newton_iters=1),
    SceneSpec("dish_towel", "Dish Towel", "deformable",
              "Fold, wring and wipe with a 30 cm cotton towel (light cloth).",
              _build_towel, robust_solver=True, newton_iters=2),
    SceneSpec("medley", "Tabletop Medley", "mixed",
              "Sphere, cup, block, peg, cube and a soft duck together.", _build_medley,
              robust_solver=True),
)


# Sub-prefabs that only duplicate a curated scene (other block colors).
_SKIP_PREFABS = {
    "box_and_blocks/block_blue.mochi_prefab",
    "box_and_blocks/block_green.mochi_prefab",
    "box_and_blocks/block_yellow.mochi_prefab",
}


def discover_prefab_scenes(roots: AssetRoots) -> list[SceneSpec]:
    """Curated prefab scenes plus any other prefab found in assets/prefabs."""
    specs: list[SceneSpec] = []
    known: set[str] = set(_SKIP_PREFABS)
    prefab_dir = roots.assets / "prefabs"
    for scene_id, name, relative, description, kwargs in _PREFAB_SCENES:
        if not (prefab_dir / relative).exists():
            continue
        known.add(relative)
        deformable = scene_id == "duck_lamp"
        # The chain's interlocked links need a smaller step to stay linked.
        fine = scene_id == "chain"
        specs.append(
            SceneSpec(
                scene_id, name, "deformable" if deformable else "rigid", description,
                _prefab_builder(relative, **kwargs),
                time_step=1.0 / 120.0 if fine else 1.0 / 60.0,
                robust_solver=deformable or fine,
            )
        )
    for path in sorted(prefab_dir.glob("*/*.mochi_prefab")):
        relative = str(path.relative_to(prefab_dir))
        if relative in known:
            continue
        scene_id = "prefab_" + path.name.split(".")[0]
        specs.append(
            SceneSpec(
                scene_id, path.name.split(".")[0].replace("_", " ").title(), "rigid",
                f"Auto-discovered prefab {relative}.", _prefab_builder(relative),
            )
        )
    return specs


def _object_builder(prefab: Path, name: str, offset=(0.0, 0.0, 0.0)):
    def build(scene: physics.Scene, roots: AssetRoots) -> None:
        add_prefab(scene, prefab, name, offset)

    return build


def _object_group_builder(prefabs: list[Path]):
    """Up to 6 objects in two rows on the work surface."""
    slots = [(-0.2, -0.09), (0.0, -0.09), (0.2, -0.09), (-0.2, 0.09), (0.0, 0.09), (0.2, 0.09)]

    def build(scene: physics.Scene, roots: AssetRoots) -> None:
        for prefab, (x, z) in zip(prefabs, slots):
            add_prefab(scene, prefab, prefab.stem.split(".")[0], (x, 0.0, z))

    return build


def discover_object_scenes(library: Path) -> list[SceneSpec]:
    """Scenes for the converted object library: one per object
    (``obj_<set>_<name>``) and groups of six (``objects_<set>_<n>``)."""
    from .objects import installed_objects

    specs: list[SceneSpec] = []
    try:
        sets = installed_objects(library)
    except OSError:
        return specs
    for set_name, prefabs in sets.items():
        for prefab in prefabs:
            name = prefab.stem.split(".")[0]
            try:
                info = json.loads((prefab.parent / "physics.json").read_text())
            except (OSError, ValueError):
                info = {}
            mu = info.get("fingertip_friction", info.get("coulomb_friction", float("nan")))
            desc = (f"{info.get('mass_kg', float('nan')) * 1000:.0f} g, fingertip friction {mu:.2f} "
                    f"({info.get('material', '?')}) from the {set_name} set.")
            pretty = re.sub(r"^\d+_", "", name).replace("_", " ").title()
            specs.append(SceneSpec(f"obj_{set_name}_{name}", f"{pretty} ({set_name.upper()})", "rigid",
                                   desc, _object_builder(prefab, name)))
        for i in range(0, len(prefabs), 6):
            group = prefabs[i:i + 6]
            names = ", ".join(re.sub(r"^\d+_", "", p.stem.split(".")[0]).replace("_", " ") for p in group)
            specs.append(SceneSpec(f"objects_{set_name}_{i // 6 + 1}",
                                   f"{set_name.upper()} objects {i // 6 + 1}", "rigid",
                                   f"{names} (measured masses).", _object_group_builder(group)))
    return specs


def scene_registry(roots: AssetRoots | None = None) -> dict[str, SceneSpec]:
    """All scenes by id, in menu order."""
    roots = roots or AssetRoots()
    specs = discover_prefab_scenes(roots) + list(_BUILTIN) + discover_object_scenes(roots.objects)
    return {spec.id: spec for spec in specs}


def build_scene(
    spec: SceneSpec,
    roots: AssetRoots,
    environment: str = "kitchen_sink",
    counter_height: float = DEFAULT_COUNTER_HEIGHT,
    time_budget: float | None = None,
) -> physics.Scene:
    """Create the scene: the physical environment (``environment``: one of
    workspace.ENVIRONMENTS, or "table" for a bare infinite tabletop plane),
    then the scene's objects on the work surface at y = 0.

    ``time_budget`` caps the Newton solve of a step at this fraction of the
    time step (wall clock), trading accuracy for real time in heavy scenes."""
    scene = physics.create_scene(spec.name)
    scene.set_gravity([0.0, -9.8, 0.0])
    if spec.robust_solver:
        _use_robust_solver(scene)
    _tune_solver(scene, spec, time_budget)
    if environment == "table":
        add_table(scene)
    else:
        _workspaces[scene.get_handle().value] = Workspace(scene, environment, counter_height)
    spec.build(scene, roots)
    return scene


_workspaces: dict[int, Workspace] = {}


def workspace_for(scene: physics.Scene) -> Workspace | None:
    return _workspaces.get(scene.get_handle().value)


def forget_scene(scene: physics.Scene) -> None:
    """Drop the per-scene registries (call before destroying the scene)."""
    handle = scene.get_handle().value
    _workspaces.pop(handle, None)
    _render_models.pop(handle, None)
