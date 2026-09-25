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

"""Physical 3D environments around the workspace.

Each environment is a set of static colliders in the physics frame (Y-up,
meters, the work surface at y = 0, the operator at +Z): a kitchen
countertop with a real sink opening and a steel basin, cabinets, a
backsplash wall and a floor; a kitchen island; a studio table. Objects can
be put in the sink, pushed off the counter and dropped on the floor.

The layout here is the single source of truth: the server sends it to the
client (``layout_message``), which builds its textured visuals from the same
numbers, so what you see is exactly what collides.

The floor follows the operator's counter height (the client measures it on
recenter and sends ``set_counter_height``); ``Workspace.set_counter_height``
moves the floor collider without rebuilding the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import superdex.physics as physics

ENV_PREFIX = "env/"
FLOOR_NAME = ENV_PREFIX + "floor"
DEFAULT_COUNTER_HEIGHT = 0.9  # [m]
ENVIRONMENTS = ("kitchen_sink", "kitchen_island", "studio")

# Contact of the environment: stone/steel/wood are all fairly grippy. The
# friction used in a pair is the geometric mean with the other actor's.
ENV_FRICTION = 0.6


@dataclass
class Box:
    """An axis-aligned static box collider (center and full size [m])."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    material: str = ""  # client material slot (countertop, steel, cabinet, ...)


@dataclass
class Plane:
    """An infinite static plane through ``point`` with outward ``normal``."""

    name: str
    normal: tuple[float, float, float]
    point: tuple[float, float, float]


@dataclass
class Layout:
    name: str
    boxes: list[Box] = field(default_factory=list)
    planes: list[Plane] = field(default_factory=list)
    # Named numbers the client needs to build matching visuals.
    params: dict = field(default_factory=dict)


def kitchen_sink_layout() -> Layout:
    """A stone countertop along a wall with an undermount stainless sink to
    the operator's right, base cabinets, upper cabinets, a faucet."""
    front, back, left, right, slab = 0.30, -0.62, -1.40, 1.40, 0.035
    # Sink opening: right of the work area, within reach of the right hand.
    sx0, sx1, sz0, sz1, depth = 0.32, 0.82, -0.46, -0.06, 0.20
    wall = 0.03  # basin wall thickness (thicker than drawn: no tunneling)
    under = 0.05  # counter slab collider thickness (extends below the visual)
    cx, cz = 0.5 * (sx0 + sx1), 0.5 * (sz0 + sz1)
    boxes = [
        # Countertop around the sink opening (top surface at y = 0).
        Box("counter_left", ((left + sx0) / 2, -under / 2, (front + back) / 2),
            (sx0 - left, under, front - back), "countertop"),
        Box("counter_right", ((sx1 + right) / 2, -under / 2, (front + back) / 2),
            (right - sx1, under, front - back), "countertop"),
        Box("counter_front", (cx, -under / 2, (sz1 + front) / 2), (sx1 - sx0, under, front - sz1), "countertop"),
        Box("counter_back", (cx, -under / 2, (back + sz0) / 2), (sx1 - sx0, under, sz0 - back), "countertop"),
        # Stainless basin: bottom and four walls, flush with the opening.
        Box("sink_bottom", (cx, -depth - wall / 2, cz), (sx1 - sx0 + 2 * wall, wall, sz1 - sz0 + 2 * wall), "steel"),
        Box("sink_wall_left", (sx0 - wall / 2, -depth / 2, cz), (wall, depth, sz1 - sz0), "steel"),
        Box("sink_wall_right", (sx1 + wall / 2, -depth / 2, cz), (wall, depth, sz1 - sz0), "steel"),
        Box("sink_wall_back", (cx, -depth / 2, sz0 - wall / 2), (sx1 - sx0 + 2 * wall, depth, wall), "steel"),
        Box("sink_wall_front", (cx, -depth / 2, sz1 + wall / 2), (sx1 - sx0 + 2 * wall, depth, wall), "steel"),
        # Base cabinets under the counter (to well below any floor height),
        # recessed 6 cm behind the counter's front edge. The part under the
        # basin starts below it.
        Box("cabinet_left", ((left + sx0 - wall) / 2, -under - 1.5, (front - 0.06 + back) / 2),
            (sx0 - wall - left, 3.0, front - 0.06 - back), "cabinet"),
        Box("cabinet_right", ((sx1 + wall + right) / 2, -under - 1.5, (front - 0.06 + back) / 2),
            (right - sx1 - wall, 3.0, front - 0.06 - back), "cabinet"),
        Box("cabinet_sink", (cx, -depth - wall - 1.5, (front - 0.06 + back) / 2),
            (sx1 - sx0 + 2 * wall, 3.0, front - 0.06 - back), "cabinet"),
        # Faucet behind the basin: base, riser, spout arm, head.
        Box("faucet_base", (cx, 0.025, sz0 - 0.07), (0.06, 0.05, 0.06), "chrome"),
        Box("faucet_riser", (cx, 0.19, sz0 - 0.07), (0.03, 0.30, 0.03), "chrome"),
        Box("faucet_arm", (cx, 0.335, sz0 + 0.03), (0.03, 0.03, 0.22), "chrome"),
        Box("faucet_head", (cx, 0.27, sz0 + 0.14), (0.035, 0.11, 0.035), "chrome"),
        # Dish-soap bottle right of the sink.
        Box("soap_bottle", (sx1 + 0.12, 0.09, sz0 + 0.02), (0.064, 0.18, 0.064), "soap"),
        # Upper cabinets left of the window.
        Box("upper_cabinets", ((left + cx - 0.45) / 2, 1.10, back + 0.17), (cx - 0.45 - left, 0.72, 0.34), "cabinet"),
        # Window sill.
        Box("sill", (cx, 0.64, back + 0.06), (0.86, 0.02, 0.12), "countertop"),
    ]
    planes = [
        Plane("backsplash", (0.0, 0.0, 1.0), (0.0, 0.0, back)),
        Plane("wall_left", (1.0, 0.0, 0.0), (left - 0.02, 0.0, 0.0)),
        Plane("wall_right", (-1.0, 0.0, 0.0), (3.2, 0.0, 0.0)),
        Plane("wall_behind", (0.0, 0.0, -1.0), (0.0, 0.0, 4.0)),
    ]
    params = {
        "front": front, "back": back, "left": left, "right": right, "slab": slab,
        "sink": [sx0, sx1, sz0, sz1], "sink_depth": depth, "faucet": [cx, sz0 - 0.07],
    }
    return Layout("kitchen_sink", boxes, planes, params)


def kitchen_island_layout(width: float = 1.2, depth: float = 0.8) -> Layout:
    """A free-standing island: things fall off every side to the floor."""
    return Layout(
        "kitchen_island",
        [
            Box("island_top", (0.0, -0.02, 0.0), (width, 0.04, depth), "butcher_block"),
            Box("island_body", (0.0, -0.04 - 1.5, 0.0), (width - 0.08, 3.0, depth - 0.1), "cabinet"),
        ],
        [],
        {"width": width, "depth": depth},
    )


def studio_layout(width: float = 1.4, depth: float = 0.9) -> Layout:
    """SuperDex Studio's table on four legs."""
    legs = [
        Box(f"leg_{i}", (x, -0.04 - 1.5, z), (0.04, 3.0, 0.04), "steel")
        for i, (x, z) in enumerate([(-0.64, -0.39), (0.64, -0.39), (-0.64, 0.39), (0.64, 0.39)])
    ]
    return Layout(
        "studio",
        [Box("tabletop", (0.0, -0.02, 0.0), (width, 0.04, depth), "tabletop")] + legs,
        [],
        {"width": width, "depth": depth},
    )


LAYOUTS = {
    "kitchen_sink": kitchen_sink_layout,
    "kitchen_island": kitchen_island_layout,
    "studio": studio_layout,
}


def layout_for(name: str) -> Layout:
    if name not in LAYOUTS:
        raise ValueError(f"unknown environment {name!r}; use one of {', '.join(LAYOUTS)}")
    return LAYOUTS[name]()


# ----------------------------------------------------------------------------
# Colliders
# ----------------------------------------------------------------------------

# Unit box triangle mesh (outward winding), scaled per collider.
_BOX_NODES = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float64
)
_BOX_TRIS = np.array(
    [
        [0, 1, 3], [0, 3, 2],  # -X
        [4, 6, 7], [4, 7, 5],  # +X
        [0, 4, 5], [0, 5, 1],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [0, 2, 6], [0, 6, 4],  # -Z
        [1, 5, 7], [1, 7, 3],  # +Z
    ],
    np.int32,
)


def box_mesh(size) -> tuple[np.ndarray, np.ndarray]:
    return _BOX_NODES * np.asarray(size, np.float64), _BOX_TRIS.copy()


def _contact() -> physics.ContactParams:
    return physics.ContactParams(coulomb_friction_coefficient=ENV_FRICTION)


def add_box(scene: physics.Scene, box: Box) -> physics.Actor:
    nodes, tris = box_mesh(box.size)
    shape = physics.create_tri_mesh_shape(
        coordinates=nodes.astype(np.float32).ravel(), connectivity=tris.ravel()
    )
    return scene.create_rigid_actor(
        name=ENV_PREFIX + box.name,
        shape=shape,
        collider_type=physics.ColliderType.BOX,
        is_static=True,
        contact=_contact(),
        world_from_local=physics.TransformRT(translation=[float(v) for v in box.center]),
    )


def _plane_transform(normal, point) -> physics.TransformRT:
    """Rotation taking +Y to ``normal``, translated to ``point``."""
    n = np.asarray(normal, np.float64)
    n /= np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(y, n)
    s, c = np.linalg.norm(axis), float(np.dot(y, n))
    if s < 1e-9:
        q = [0.0, 0.0, 0.0, 1.0] if c > 0 else [1.0, 0.0, 0.0, 0.0]
    else:
        axis /= s
        half = 0.5 * math.atan2(s, c)
        q = list(axis * math.sin(half)) + [math.cos(half)]
    return physics.TransformRT(
        rotation=physics.Quaternion(*[float(v) for v in q]),
        translation=[float(v) for v in point],
    )


def add_plane(scene: physics.Scene, plane: Plane) -> physics.Actor:
    return scene.create_rigid_actor(
        name=ENV_PREFIX + plane.name,
        shape=physics.create_plane_shape(normal=[0, 1, 0], distance=0.0),
        is_static=True,
        contact=_contact(),
        world_from_local=_plane_transform(plane.normal, plane.point),
    )


class Workspace:
    """The static environment actors of one scene."""

    def __init__(self, scene: physics.Scene, name: str, counter_height: float = DEFAULT_COUNTER_HEIGHT):
        self.layout = layout_for(name)
        self.name = name
        self.actors: list[physics.Actor] = []
        for box in self.layout.boxes:
            self.actors.append(add_box(scene, box))
        for plane in self.layout.planes:
            self.actors.append(add_plane(scene, plane))
        self.floor = scene.create_rigid_actor(
            name=FLOOR_NAME,
            shape=physics.create_plane_shape(normal=[0, 1, 0], distance=0.0),
            is_static=True,
            contact=_contact(),
        )
        self.actors.append(self.floor)
        self.counter_height = counter_height
        self.set_counter_height(counter_height)

    def set_counter_height(self, height: float) -> None:
        """Put the floor ``height`` below the work surface (clamped to 0.3-1.6 m)."""
        self.counter_height = float(min(max(height, 0.3), 1.6))
        self.floor.set_root_transform(physics.TransformRT(translation=[0.0, -self.counter_height, 0.0]))

    def message(self) -> dict:
        """The layout for the client (it builds matching visuals)."""
        return layout_message(self.layout, self.counter_height)


def layout_message(layout: Layout, counter_height: float = DEFAULT_COUNTER_HEIGHT) -> dict:
    return {
        "name": layout.name,
        "counter_height": counter_height,
        "params": layout.params,
        "boxes": [
            {"name": ENV_PREFIX + b.name, "center": list(b.center), "size": list(b.size), "material": b.material}
            for b in layout.boxes
        ],
        "planes": [
            {"name": ENV_PREFIX + p.name, "normal": list(p.normal), "point": list(p.point)}
            for p in layout.planes
        ],
    }


def is_environment_actor(name: str) -> bool:
    return name.startswith(ENV_PREFIX)
