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

"""Physical environments, hand-size calibration, unstick and replay."""

import numpy as np
import pytest

from superdex_quest_teleop import hand_skeleton as hs
from superdex_quest_teleop.calibration import calibrate, hand_length
from superdex_quest_teleop.scenes import build_scene, forget_scene, scene_registry, workspace_for
from superdex_quest_teleop.workspace import ENVIRONMENTS, kitchen_sink_layout


def _drop(scene, actor, position, steps=120, dt=1 / 60):
    import superdex.physics as physics

    actor.set_root_transform(physics.TransformRT(translation=list(position)))
    for _ in range(steps):
        scene.step(dt)
    return actor.get_aabb_world()


def _rigid(scene):
    actors = []
    scene.for_each_actor(lambda a: actors.append(a))
    return next(a for a in actors if not a.is_static())


def test_kitchen_sink_is_physical(roots):
    import superdex.physics as physics

    layout = kitchen_sink_layout()
    sx0, sx1, sz0, sz1 = layout.params["sink"]
    depth = layout.params["sink_depth"]
    scene = build_scene(scene_registry(roots)["cube"], roots, "kitchen_sink", counter_height=0.85)
    try:
        cube = _rigid(scene)
        # Into the sink: rests on the basin bottom, inside the opening.
        box = _drop(scene, cube, (0.5 * (sx0 + sx1) - 0.03, 0.15, 0.5 * (sz0 + sz1) - 0.03))
        assert box.min[1] == pytest.approx(-depth, abs=0.01)
        assert sx0 - 1e-3 < box.min[0] and box.max[0] < sx1 + 1e-3
        # Off the front edge: falls to the floor 0.85 m below the counter.
        box = _drop(scene, cube, (-0.03, 0.1, layout.params["front"] + 0.12), steps=150)
        assert box.min[1] == pytest.approx(-0.85, abs=0.01)
        # The floor follows the operator's counter height.
        workspace_for(scene).set_counter_height(0.95)
        box = _drop(scene, cube, (-0.03, 0.1, layout.params["front"] + 0.12), steps=150)
        assert box.min[1] == pytest.approx(-0.95, abs=0.01)
        # On the counter: rests at y = 0.
        box = _drop(scene, cube, (-0.03, 0.05, -0.03))
        assert box.min[1] == pytest.approx(0.0, abs=0.005)
    finally:
        forget_scene(scene)
        physics.destroy_scene(scene)


@pytest.mark.parametrize("environment", [e for e in ENVIRONMENTS if e != "kitchen_sink"])
def test_other_environments_have_an_edge(roots, environment):
    import superdex.physics as physics

    scene = build_scene(scene_registry(roots)["cube"], roots, environment, counter_height=0.9)
    try:
        cube = _rigid(scene)
        assert _drop(scene, cube, (-0.03, 0.05, -0.03)).min[1] == pytest.approx(0.0, abs=0.005)
        # Past the side of the island / studio table: onto the floor.
        assert _drop(scene, cube, (0.9, 0.1, -0.03), steps=150).min[1] == pytest.approx(-0.9, abs=0.01)
    finally:
        forget_scene(scene)
        physics.destroy_scene(scene)


def test_hand_scale_calibration(roots):
    import superdex.physics as physics
    from superdex_quest_teleop.hand_rig import HandUnit

    scene = build_scene(scene_registry(roots)["cube"], roots)
    try:
        refs = {}
        for side, x in (("left", -0.2), ("right", 0.2)):
            hand = HandUnit(scene, roots, side, (x, 0.3, 0.2))
            refs[side] = hand.retargeter.synthesize_skeleton(np.zeros(hand.kinematics.num_dofs))
        # A user whose hands are 7% larger than the Meta XR hand asset.
        scale, report = calibrate({s: 1.07 * refs[s] + [0.1, 0.2, 0.3] for s in refs}, refs)
        assert scale == pytest.approx(1.07, abs=1e-6)
        assert report["hand_length_cm"]["right"] == pytest.approx(hand_length(refs["right"]) * 107, abs=0.1)
        # One hand is enough; absurd measurements are clamped.
        assert calibrate({"right": 3.0 * refs["right"]}, refs)[0] == pytest.approx(1.35)
        # The scaled bot is uniformly larger: skeleton, collision shapes.
        big = HandUnit(scene, roots, "right", (0.0, 0.3, -0.2), scale=1.2)
        skeleton = big.retargeter.synthesize_skeleton(np.zeros(big.kinematics.num_dofs))
        assert hand_length(skeleton) == pytest.approx(1.2 * hand_length(refs["right"]), rel=1e-4)
        lo = np.min([list(a.get_aabb_world().min) for a in big.link_actors], axis=0)
        hi = np.max([list(a.get_aabb_world().max) for a in big.link_actors], axis=0)
        assert np.max(hi - lo) > 1.15 * 0.2  # the unit hand spans ~0.21 m
    finally:
        forget_scene(scene)
        physics.destroy_scene(scene)


def test_unstick_passes_through_and_recovers(roots):
    """A tracked hand far below the counter (the physical hand stuck on it)
    stops colliding after UNSTICK_AFTER, then collides again once caught up."""
    from superdex_quest_teleop import hand_rig
    from superdex_quest_teleop.session import TeleopSession
    from superdex_quest_teleop.synthetic import grasp_pose

    session = TeleopSession(scene_registry(roots)["cube"], roots)
    try:
        hand = session.hands["right"]
        kin = hand.kinematics
        rot = hand._neutral_rotation()

        def frame(y):
            w = np.eye(4)
            w[:3, :3] = rot
            w[:3, 3] = [-0.2, y, 0.1]  # over plain counter (not the sink)
            return hs.HandFrame(True, hand.retargeter.synthesize_skeleton(grasp_pose(kin, 0.1), w))

        def run(y, seconds):
            for _ in range(int(seconds / session.time_step)):
                session.set_input({"right": frame(y), "left": hs.HandFrame.untracked()})
                session.step(with_contacts=False)

        run(0.15, 1.0)
        assert not hand.passing_through
        run(-0.25, 0.3)  # pushed 25 cm into the counter: blocked, not yet passing
        assert not hand.passing_through
        assert hand.link_poses()[0][1] > -0.02
        run(-0.25, 1.0)
        assert hand.passing_through  # gave up and went through
        run(0.15, 1.5)  # back above the counter and caught up
        assert not hand.passing_through
        assert hand.link_poses()[0][1] == pytest.approx(0.15, abs=0.02)
        assert hand_rig.UNSTICK_DISTANCE > 0.05
    finally:
        session.close()


def test_rate_limit_ignores_tracking_jumps(roots):
    from superdex_quest_teleop.session import TeleopSession

    session = TeleopSession(scene_registry(roots)["cube"], roots)
    try:
        hand = session.hands["right"]
        rot = hand._neutral_rotation()

        def frame(x):
            w = np.eye(4)
            w[:3, :3] = rot
            w[:3, 3] = [x, 0.2, 0.1]
            return hs.HandFrame(True, hand.retargeter.synthesize_skeleton(np.zeros(hand.kinematics.num_dofs), w))

        for _ in range(30):
            session.set_input({"right": frame(0.0), "left": hs.HandFrame.untracked()})
            session.step(with_contacts=False)
        # A 30 cm glitch for one frame moves the command by at most v_max * dt.
        session.set_input({"right": frame(0.3), "left": hs.HandFrame.untracked()})
        session.step(with_contacts=False)
        from superdex_quest_teleop.hand_rig import MAX_WRIST_SPEED

        assert hand._cmd_root[0, 3] == pytest.approx(MAX_WRIST_SPEED * session.time_step, abs=1e-6)
    finally:
        session.close()


def test_replay_streams_recorded_frames(roots, tmp_path):
    """Record a few scripted steps, then play them back through the replay
    runner: frames decode to the recorded poses."""
    import json
    import struct

    from superdex_quest_teleop.replay import Episode, ReplayRunner
    from superdex_quest_teleop.server import ServerConfig
    from superdex_quest_teleop.session import TeleopSession
    from superdex_quest_teleop.synthetic import ScriptedGrasp

    session = TeleopSession(scene_registry(roots)["kitchen_sponge"], roots)
    grasp = ScriptedGrasp("right", target=(0.05, 0, 0), object_top=0.032, kinematics=session.hands["right"].kinematics)
    session.start_recording(tmp_path)
    for k in range(40):
        session.set_input({"right": grasp.frame(k * session.time_step), "left": hs.HandFrame.untracked()})
        session.step()
    path, steps = session.stop_recording()
    recorded_pose = session.gather(with_contacts=False)["pose"]
    session.close()
    assert steps == 40

    published = []
    episode = Episode(path)
    runner = ReplayRunner(ServerConfig(environment_auto=False), roots,
                          lambda kind, payload: published.append((kind, payload)), episode)
    runner._build()
    try:
        geometry = runner.geometry_message()
        assert geometry["environment"]["name"] == "kitchen_sink"
        sponge = next(a for a in geometry["actors"] if a["name"] == "sponge")
        assert len(sponge["positions"]) % 3 == 0 and np.isfinite(sponge["positions"]).all()
        frame = runner._frame(39)
        length = struct.unpack("<I", frame[:4])[0]
        header = json.loads(frame[4:4 + length])
        assert header["step"] == 39 and header["num_actors"] == len(recorded_pose)
        offset = 4 + length + (-(4 + length)) % 4
        pose = np.frombuffer(frame[offset:offset + 4 * 7 * len(recorded_pose)], "<f4").reshape(-1, 7)
        with_names = [a["name"] for a in geometry["actors"]]
        i = with_names.index("sponge")
        assert np.allclose(pose[i], episode.file["actors/pose"][39][episode.actor_names.index("sponge")])
        assert header["hand_joints"] == ["left", "right"]
    finally:
        runner.session.close()
        episode.close()
