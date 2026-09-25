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

"""Replay a recorded episode in the web viewer.

    python -m superdex_quest_teleop.replay recordings/kitchen_sponge_20260925_101500.h5

opens the same 3D view as live teleoperation (desktop browser, or the Quest
for an immersive look) and plays the episode back from the file: objects,
deforming bodies, the hands, and the recorded contact points with their
force vectors. Play/pause, scrub the timeline, change the speed, step frame
by frame. Nothing is simulated: the scene is only rebuilt to get its
meshes, everything that moves comes from the file.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import time
from pathlib import Path

import h5py
import numpy as np

from .scenes import AssetRoots, default_repo_root, scene_registry
from .server import PhysicsRunner, ServerConfig, TeleopServer, _jsonable_actor
from .session import TeleopSession

log = logging.getLogger("superdex_quest_teleop")

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


class Episode:
    """Random access to one recorded episode."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        f = self.file
        self.scene_id = str(f.attrs["scene_id"])
        self.time_step = float(f.attrs["time_step"])
        self.metadata = json.loads(f.attrs.get("metadata", "{}"))
        self.num_steps = int(f["actors/pose"].shape[0])
        self.actor_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["actors/name"][:]]
        self.sim_time = f["steps/sim_time"][:]
        self.sides = [s for s in ("left", "right") if f"hands/{s}/link_pose" in f]
        self.offsets = f["contacts/step_offset"][:]
        self.deformables = []
        for key in sorted(f["deformables"].keys(), key=int) if "deformables" in f else []:
            grp = f[f"deformables/{key}"]
            self.deformables.append({
                "name": str(grp.attrs["name"]),
                "positions": grp["surface_positions"],
                "triangles": grp["surface_triangles"][:] if "surface_triangles" in grp else None,
            })

    def contacts(self, step: int) -> dict[str, np.ndarray]:
        s = slice(int(self.offsets[step]), int(self.offsets[step + 1]))
        f = self.file
        return {
            "force": f["contacts/force"][s],
            "pos_owner": f["contacts/pos_owner"][s],
        }

    def close(self) -> None:
        self.file.close()


class ReplayRunner(PhysicsRunner):
    """Stands in for the physics thread: streams recorded frames."""

    def __init__(self, config: ServerConfig, roots: AssetRoots, publish, episode: Episode) -> None:
        super().__init__(config, roots, publish)
        self.episode = episode
        self.cursor = 0
        self.playing = True
        self.speed = 1.0
        self.loop = True

    # Thread-safe entry points ------------------------------------------------

    def set_hands(self, hands, head) -> None:  # live input is ignored
        pass

    def scene_list(self) -> list[dict]:
        spec = self.registry.get(self.episode.scene_id)
        name = spec.name if spec else self.episode.scene_id
        return [{"id": self.episode.scene_id, "name": f"Replay: {name}", "category": "replay",
                 "description": str(self.episode.path), "time_step": self.episode.time_step}]

    def replay_info(self) -> dict:
        e = self.episode
        return {
            "file": str(e.path), "num_steps": e.num_steps, "time_step": e.time_step,
            "duration": float(e.sim_time[-1] - e.sim_time[0]) if e.num_steps else 0.0,
            "speeds": list(SPEEDS),
        }

    def status(self) -> dict:
        return {
            "type": "status", "scene": self.episode.scene_id, "recording": False, "file": None,
            "contact_mode": self.episode.metadata.get("contact_mode"),
            "environment": self.environment,
            "replay": {**self.replay_info(), "cursor": self.cursor, "playing": self.playing,
                       "speed": self.speed},
        }

    # Physics thread ----------------------------------------------------------

    def run(self) -> None:
        import superdex.physics as physics

        self._owns_physics = not physics.is_initialized()
        if self._owns_physics:
            physics.initialize(num_worker_threads=1)
        try:
            self._build()
            self.ready.set()
            self._play()
        finally:
            if self.session is not None:
                self.session.close()
                self.session = None
            if self._owns_physics:
                physics.shutdown()

    def _build(self) -> None:
        e = self.episode
        if e.scene_id not in self.registry:
            raise SystemExit(f"{e.path}: scene {e.scene_id!r} is not in this SuperDex checkout")
        spec = self.registry[e.scene_id]
        # Episodes before the physical environments were recorded on a bare table.
        self.environment = e.metadata.get("environment", "table")
        height = e.metadata.get("counter_height")
        session = TeleopSession(
            spec, self.roots,
            contact_mode=e.metadata.get("contact_mode", "hand"),
            environment=self.environment,
            **({"counter_height": float(height)} if isinstance(height, (int, float)) and np.isfinite(height) else {}),
        )
        names = [r.name for r in session.actors]
        index = {n: i for i, n in enumerate(e.actor_names)}
        missing = [n for n in names if n not in index and not n.startswith("env/") and n != "table"]
        if missing:
            log.warning("actors not in the recording (shown at rest): %s", ", ".join(missing[:5]))
        # Recorded actor index for each session actor (-1: not recorded).
        self._actor_map = np.array([index.get(n, -1) for n in names])
        self._rest_pose = session.gather(with_contacts=False)["pose"]
        # Deformables: show the recorded surface (its vertex count may differ
        # from what this version of the session would stream).
        rec_def = {d["name"]: d for d in e.deformables}
        self._deformables = [rec_def.get(r.name) for r in session.deformables]
        actors = []
        for entry in session.geometry():
            d = rec_def.get(entry["name"]) if entry["deformable"] else None
            if d is not None and d["triangles"] is not None:
                entry["mesh"] = "surface"
                entry["positions"] = np.asarray(d["positions"][0], np.float32).reshape(-1)
                entry["indices"] = np.asarray(d["triangles"], np.uint32).reshape(-1)
            actors.append(_jsonable_actor(entry))
        self._geometry_message = {
            "type": "geometry",
            "scene": {"id": spec.id, "name": f"Replay: {spec.name}",
                      "description": f"{e.path.name}: {e.num_steps} steps", "time_step": e.time_step},
            "actors": actors,
            "environment": session.environment_message(),
        }
        self.session = session
        self.publish("geometry", self._geometry_message)
        self.publish("status", self.status())
        log.info("replaying %s: %s, %d steps (%.1f s)", e.path, spec.name, e.num_steps,
                 e.num_steps * e.time_step)

    def _play(self) -> None:
        e = self.episode
        last = time.monotonic()
        accum = 0.0
        next_stream = 0.0
        next_status = 0.0
        dirty = True
        while self.running:
            self._drain_commands()
            now = time.monotonic()
            if self.playing and e.num_steps:
                accum += (now - last) * self.speed
                advance = int(accum / e.time_step)
                if advance:
                    accum -= advance * e.time_step
                    self.cursor += advance
                    if self.cursor >= e.num_steps:
                        self.cursor = 0 if self.loop else e.num_steps - 1
                        self.playing = self.loop
                    dirty = True
            last = now
            # Paused: repeat the frame now and then so newly opened viewers get it.
            if not dirty and not self.playing and now >= next_stream + 0.5:
                dirty = True
            if dirty and now >= next_stream:
                self.publish("frame", self._frame(self.cursor))
                next_stream = now + 1.0 / self.config.stream_hz
                dirty = False
            if now >= next_status:
                self.publish("status", self.status())
                next_status = now + 0.5
            time.sleep(0.004)

    def _frame(self, step: int) -> bytes:
        e, s = self.episode, self.session
        f = e.file
        recorded = f["actors/pose"][step]
        pose = self._rest_pose.copy()
        ok = self._actor_map >= 0
        pose[ok] = recorded[self._actor_map[ok]]
        deformables = []
        for r, d in zip(s.deformables, self._deformables):
            positions = d["positions"][step] if d is not None else np.zeros((0, 3), np.float32)
            deformables.append({"surface_positions": np.asarray(positions, np.float32)})
        hands = {}
        for side, hand in s.hands.items():
            if side not in e.sides:
                continue
            link_pose = f[f"hands/{side}/link_pose"][step]
            hands[side] = {
                "tracked": bool(f[f"hands/{side}/tracked"][step]),
                "display_joints": hand.display.joints(link_pose),
            }
        s.last_contacts = e.contacts(step)
        head = f["head/pose"][step] if "head/pose" in f else None
        data = {
            "step": step, "sim_time": float(e.sim_time[step]), "pose": pose,
            "deformables": deformables, "hands": hands, "head_pose": head,
        }
        self.rtf = self.speed if self.playing else 0.0
        return self._encode_frame(s, data)

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            name = cmd.get("cmd")
            n = self.episode.num_steps
            if name == "stop":
                self.running = False
            elif name in ("replay_toggle", "record_toggle"):
                if not self.playing and self.cursor >= n - 1:
                    self.cursor = 0
                self.playing = not self.playing
            elif name == "replay_play":
                self.playing = True
            elif name == "replay_pause":
                self.playing = False
            elif name == "replay_seek":
                self.cursor = int(min(max(int(cmd.get("step", 0)), 0), max(n - 1, 0)))
                self.publish("frame", self._frame(self.cursor))
            elif name == "replay_step":
                self.playing = False
                self.cursor = int(min(max(self.cursor + int(cmd.get("delta", 1)), 0), max(n - 1, 0)))
                self.publish("frame", self._frame(self.cursor))
            elif name == "replay_speed":
                self.speed = float(min(max(float(cmd.get("speed", 1.0)), 0.01), 10.0))
            elif name in ("reset", "load_scene", "next_scene", "prev_scene"):
                self.cursor = 0
                self.publish("frame", self._frame(self.cursor))
            else:
                continue  # recording, environment and hand commands don't apply
            self.publish("status", self.status())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m superdex_quest_teleop.replay",
        description="Play a recorded teleoperation episode (.h5) in the web viewer.",
    )
    parser.add_argument("episode", type=Path, help="episode file written by the teleop server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port of the viewer")
    parser.add_argument("--https-port", type=int, default=8443,
                        help="HTTPS port (to watch the replay in the Quest over Wi-Fi)")
    parser.add_argument("--no-https", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--repo", type=Path, default=None, help="SuperDex checkout (default: auto)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.episode.is_file():
        parser.error(f"no such file: {args.episode}")

    episode = Episode(args.episode)
    roots = AssetRoots(args.repo.resolve() if args.repo else default_repo_root())
    if episode.scene_id not in scene_registry(roots):
        parser.error(f"scene {episode.scene_id!r} of this episode is not in {roots.repo}")
    config = ServerConfig(
        http_port=args.port, https_port=args.https_port, https=not args.no_https,
        environment_auto=False, scene=episode.scene_id,
    )
    server = TeleopServer(
        config, roots,
        runner_factory=lambda cfg, r, publish: ReplayRunner(cfg, r, publish, episode),
    )
    server.runner.speed = args.speed
    print(f"Replaying {args.episode} ({episode.num_steps} steps, {episode.num_steps * episode.time_step:.1f} s)")
    print(f"  viewer : http://localhost:{args.port}/")
    print("  keys   : Space play/pause, ←/→ step, timeline slider, C contacts, V cam view")
    print("  quit   : Ctrl-C")
    try:
        server.run()
    finally:
        episode.close()


if __name__ == "__main__":
    main()
