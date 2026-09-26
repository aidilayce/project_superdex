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

"""Object (SDFormat/URDF -> prefab) and room (glTF -> environment) import,
on synthetic inputs (no downloads)."""

import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")
pytest.importorskip("skimage")


def _box(ext, center):
    b = trimesh.creation.box(extents=ext)
    b.apply_translation(center)
    return b


SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <link name="base_link">
      <inertial>
        <mass>{mass}</mass>
        <pose>0 0 0.01 0 0 0</pose>
        <inertia><ixx>1e-4</ixx><iyy>2e-4</iyy><izz>3e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="visual"><geometry><mesh><uri>{visual}</uri></mesh></geometry></visual>
      <collision name="c0">
        <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        <geometry><box><size>0.06 0.04 0.1</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def test_sdf_object_to_prefab(tmp_path, superdex_initialized):
    import superdex.physics as physics

    from superdex_quest_teleop.objects import MATERIALS, actor_friction, convert, installed_objects
    from superdex_quest_teleop.scenes import AssetRoots, add_prefab, build_scene, discover_object_scenes

    # A box whose SDF primitive fits its visual mesh, and a "bottle" (a box
    # primitive 3 cm shorter than the visual mesh, like Drake's mustard).
    _box((0.06, 0.04, 0.1), (0, 0, 0)).export(tmp_path / "tin.obj")
    bottle = trimesh.util.concatenate([_box((0.06, 0.04, 0.1), (0, 0, 0)),
                                       trimesh.creation.cylinder(radius=0.012, height=0.03)
                                       .apply_translation((0, 0, 0.065))])
    bottle.export(tmp_path / "bottle.obj")
    # SceneSmith writes same-material friction: 0.74 is steel on steel.
    (tmp_path / "tin.sdf").write_text(SDF.format(name="steel_tin", mass=0.25, visual="tin.obj", mu=0.74))
    (tmp_path / "bottle.sdf").write_text(SDF.format(name="bottle", mass=0.4, visual="bottle.obj", mu=0.35))
    library = tmp_path / "library"
    tin = convert(tmp_path / "tin.sdf", library / "test")
    bot = convert(tmp_path / "bottle.sdf", library / "test")

    info = json.loads((tin.parent / "physics.json").read_text())
    assert info["material"] == "metal" and info["fingertip_friction"] == MATERIALS["metal"]
    assert info["coulomb_friction"] == pytest.approx(actor_friction(MATERIALS["metal"]))
    assert info["collision"] == "box primitive" and info["collider"] == "BOX"
    binfo = json.loads((bot.parent / "physics.json").read_text())
    assert binfo["material"] == "plastic"  # name "bottle" disambiguates SceneSmith's 0.35
    assert "remesh" in binfo["collision"] or binfo["collision"].startswith("closed")
    assert binfo["extent_m"][2] == pytest.approx(0.13, abs=0.006)  # the neck is kept
    assert (tin.parent / "render" / "steel_tin.glb").exists()
    assert set(installed_objects(library)) == {"test"}

    # In SuperDex: measured mass and inertia, material friction, rests on the counter.
    scene = build_scene(next(s for s in discover_object_scenes(library) if s.id == "obj_test_steel_tin"),
                        AssetRoots())
    try:
        actors = []
        scene.for_each_actor(lambda a: actors.append(a))
        tin_actor = next(a for a in actors if not a.is_static())
        assert tin_actor.get_mass() == pytest.approx(0.25, rel=1e-4)
        mu = tin_actor.get_contact_params().coulomb_friction_coefficient
        assert np.sqrt(mu * 1.0) == pytest.approx(MATERIALS["metal"], rel=1e-4)  # finger contact
        for _ in range(90):
            scene.step(1 / 60)
        assert abs(tin_actor.get_aabb_world().min[1]) < 0.003
    finally:
        physics.destroy_scene(scene)
    assert add_prefab  # (re-exported helper used by the object scenes)


def test_urdf_mass_and_generic_name(tmp_path, superdex_initialized):
    from superdex_quest_teleop.objects import parse_urdf

    folder = tmp_path / "YcbThing"
    folder.mkdir()
    _box((0.05, 0.05, 0.05), (0, 0, 0)).export(folder / "mesh.obj")
    (folder / "model.urdf").write_text(
        '<robot name="model.urdf"><link name="base"><inertial><origin xyz="0 0 0.01"/>'
        '<mass value=".066"/><inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>'
        '</inertial><visual><geometry><mesh filename="mesh.obj"/></geometry></visual>'
        '<collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link></robot>')
    obj = parse_urdf(folder / "model.urdf")
    assert obj.name == "YcbThing" and obj.mass == pytest.approx(0.066)
    assert np.allclose(obj.center_of_mass, [0, 0, 0.01])


def _room(path):
    """5 x 4 x 2.6 m room in centimetres: a desk against the back wall, a
    coffee table, a sofa, a bookshelf."""
    parts = {}
    shell = trimesh.creation.box(extents=(5, 2.6, 4))
    shell.apply_translation((0, 1.3, 0))
    shell.invert()
    parts["room_shell"] = shell
    parts["desk"] = trimesh.util.concatenate(
        [_box((1.4, 0.04, 0.7), (0.5, 0.73, -1.6))]
        + [_box((0.05, 0.71, 0.05), (0.5 + sx * 0.65, 0.355, -1.6 + sz * 0.3)) for sx in (-1, 1) for sz in (-1, 1)])
    parts["coffee_table"] = _box((1.0, 0.04, 0.6), (-1.2, 0.43, 0.8))
    parts["sofa"] = trimesh.util.concatenate([_box((2.0, 0.45, 0.9), (-1.2, 0.225, 1.5)),
                                              _box((2.0, 0.45, 0.2), (-1.2, 0.675, 1.85))])
    scene = trimesh.Scene()
    cm = np.diag([100.0, 100.0, 100.0, 1.0])
    for name, mesh in parts.items():
        mesh = mesh.copy()
        mesh.apply_transform(cm)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(str(path))


def test_room_import_and_physics(tmp_path, monkeypatch, superdex_initialized):
    import superdex.physics as physics

    from superdex_quest_teleop import rooms
    from superdex_quest_teleop.scenes import AssetRoots, build_scene, forget_scene, scene_registry, workspace_for

    monkeypatch.setattr(rooms, "ROOMS_DIR", tmp_path / "rooms")
    _room(tmp_path / "room.glb")
    rooms.import_room("den", tmp_path / "room.glb", root=tmp_path / "rooms")
    info = rooms.installed_rooms(tmp_path / "rooms")["den"]
    assert info["scale"] == 0.01 and info["room_height_m"] == pytest.approx(2.6, abs=0.05)
    assert info["counter_height"] == pytest.approx(0.75, abs=0.011)  # the desk, not the coffee table
    assert info["surface"]["size"][0] == pytest.approx(1.4, abs=0.06)
    T = np.asarray(info["physics_from_model"])
    # The operator faces the desk from the room (the wall is behind it: -Z).
    desk_center = T @ [50.0, 75.0, -160.0, 1.0]
    wall = T @ [50.0, 75.0, -200.0, 1.0]
    assert np.allclose(desk_center[:3], 0.0, atol=0.03) and wall[2] < -0.3

    roots = AssetRoots()
    scene = build_scene(scene_registry(roots)["cube"], roots, "den")
    try:
        ws = workspace_for(scene)
        ws.set_counter_height(0.9)  # rooms keep their modeled height
        assert ws.counter_height == pytest.approx(info["counter_height"])
        msg = ws.message()
        assert msg["kind"] == "room" and msg["model_url"].startswith("/rooms/den/")
        actors = []
        scene.for_each_actor(lambda a: actors.append(a))
        cube = next(a for a in actors if not a.is_static())

        def drop(pos, steps=150):
            cube.set_root_transform(physics.TransformRT(translation=list(pos)))
            for _ in range(steps):
                scene.step(1 / 60)
            return cube.get_aabb_world()

        assert drop((0.0, 0.05, 0.0)).min[1] == pytest.approx(0.0, abs=0.004)  # on the desk
        assert drop((0.0, 0.05, 0.5)).min[1] == pytest.approx(-ws.counter_height, abs=0.01)  # floor
        box = drop((0.0, 0.3, -0.36))  # between the desk's back edge and the wall
        assert box.min[2] > wall[2] - 0.02  # stopped by the wall, not through it
    finally:
        forget_scene(scene)
        physics.destroy_scene(scene)
