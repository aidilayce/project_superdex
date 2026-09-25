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

    def __init__(self, config: ServerConfig, roots: AssetRoots, publish, episode: Episode,
                 export_path: Path | None = None) -> None:
        super().__init__(config, roots, publish)
        self.episode = episode
        self.cursor = 0
        self.playing = True
        self.speed = 1.0
        self.loop = True
        # MP4 export: the viewer renders frames, this encodes them.
        self.export_path = export_path
        self.export_dir = (export_path or episode.path).resolve().parent
        self._writer = None

    # MP4 export (called from the web server's event loop, in order) ----------

    def export_command(self, message: dict) -> dict:
        from .video import Mp4Writer

        action = message.get("action")
        try:
            if action == "start":
                if self._writer is not None:
                    self._writer.abort()
                path = self.export_path or _free_name(self.episode.path.with_suffix(".mp4"))
                self._writer = Mp4Writer(path, float(message.get("fps", 30)))
                log.info("exporting %s (%s frames at %s fps)", path, message.get("frames"), message.get("fps"))
                return {"ok": True, "action": "started", "path": str(path)}
            if self._writer is None:
                return {"ok": False, "action": action, "error": "no export in progress"}
            writer, self._writer = self._writer, None
            if action == "abort":
                writer.abort()
                return {"ok": True, "action": "aborted"}
            path = writer.close()
            log.info("exported %s (%d frames, %.1f s)", path, writer.frames, writer.frames / writer.fps)
            return {"ok": True, "action": "finished", "path": str(path), "frames": writer.frames,
                    "url": f"/exports/{path.name}" if path.parent == self.export_dir else None}
        except Exception as exc:  # noqa: BLE001 - reported to the viewer
            log.exception("export %s failed", action)
            self._writer = None
            return {"ok": False, "action": action, "error": str(exc)}

    def export_frame(self, jpeg: bytes) -> None:
        if self._writer is not None:
            self._writer.write(jpeg)

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


def export_mp4(episode: Episode, roots: AssetRoots, out: Path, args, size: tuple[int, int]) -> Path:
    """Render ``episode`` to ``out`` with a headless browser running the viewer."""
    import asyncio
    import socket
    import sys
    import threading
    import urllib.parse

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("--mp4 renders with a headless browser: pip install playwright imageio-ffmpeg, "
                         "then `python -m playwright install chromium` (or pass --browser /path/to/chrome)")
    from .video import find_ffmpeg

    find_ffmpeg()  # fail early with the install hint
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = ServerConfig(host="127.0.0.1", http_port=port, https=False, environment_auto=False,
                          scene=episode.scene_id)
    server = TeleopServer(
        config, roots,
        runner_factory=lambda cfg, r, publish: ReplayRunner(cfg, r, publish, episode, export_path=out),
    )
    server.runner.playing = False
    stop_holder: dict = {}

    def serve() -> None:
        async def main() -> None:
            stop_holder["stop"] = asyncio.Event()
            stop_holder["loop"] = asyncio.get_running_loop()
            await server.serve(stop_holder["stop"])

        asyncio.run(main())

    thread = threading.Thread(target=serve, name="replay-server", daemon=True)
    thread.start()
    if not server.runner.ready.wait(120):
        raise SystemExit("the replay server did not start")

    query = {
        "export": "1", "fps": f"{args.fps:g}", "speed": f"{args.speed:g}", "start": f"{args.start:g}",
        "view": args.view, "cam": args.camera, "contacts": "0" if args.no_contacts else "1",
        "overlay": "0" if args.no_overlay else "1",
    }
    if args.end is not None:
        query["end"] = f"{args.end:g}"
    url = f"http://127.0.0.1:{port}/?{urllib.parse.urlencode(query)}"
    duration = ((args.end if args.end is not None else episode.num_steps * episode.time_step) - args.start)
    print(f"rendering {out} ({size[0]}x{size[1]}, {args.fps:g} fps, {max(duration, 0) / args.speed:.1f} s of video)")
    gl_args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] \
        if sys.platform.startswith("linux") else []
    try:
        with sync_playwright() as pw:
            launch = dict(headless=True, args=gl_args + ["--no-proxy-server"])
            if args.browser:
                browser = pw.chromium.launch(executable_path=args.browser, **launch)
            else:
                try:
                    browser = pw.chromium.launch(**launch)
                except Exception:  # noqa: BLE001 - no Playwright Chromium: try Google Chrome
                    browser = pw.chromium.launch(channel="chrome", **launch)
            page = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1)
            def console(message) -> None:
                if message.text.startswith("export "):
                    _progress(message.text)
                elif getattr(args, "verbose", False):
                    print(f"[page {message.type}] {message.text}")

            page.on("console", console)
            page.on("pageerror", lambda e: log.warning("page error: %s", e))
            page.goto(url)
            result = None
            while result is None:
                page.wait_for_timeout(500)
                result = page.evaluate("window.__exportResult || null")
            browser.close()
    finally:
        print()
        loop = stop_holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(stop_holder["stop"].set)
        thread.join(60)
    if not result.get("ok"):
        raise SystemExit(f"export failed: {result.get('error')}")
    return Path(result["path"])


def _progress(text: str) -> None:
    done, total = (int(v) for v in text.split()[1].split("/"))
    bar = "#" * int(30 * done / total)
    print(f"\r  [{bar:<30}] {done}/{total} frames", end="", flush=True)


def _free_name(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


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
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed (with --mp4: video speed, e.g. 0.25 for slow motion)")
    parser.add_argument("--repo", type=Path, default=None, help="SuperDex checkout (default: auto)")
    video = parser.add_argument_group("MP4 export (renders headless, no viewer needed)")
    video.add_argument("--mp4", nargs="?", const="", default=None, metavar="OUT.mp4",
                       help="render the episode to an MP4 and exit (default name: the episode's, .mp4)")
    video.add_argument("--fps", type=float, default=30.0)
    video.add_argument("--size", default="1920x1080", help="video size WxH")
    video.add_argument("--view", choices=("orbit", "cam"), default="orbit",
                       help="orbit: a fixed camera over the counter (see --camera); "
                       "cam: through the operator's eyes (recorded headset pose)")
    video.add_argument("--camera", default="0.0,1.55,0.75,0.05,0.9,-0.1", metavar="PX,PY,PZ,TX,TY,TZ",
                       help="orbit camera position and target in room coordinates (counter top at y=0.9)")
    video.add_argument("--start", type=float, default=0.0, help="start time [s]")
    video.add_argument("--end", type=float, default=None, help="end time [s] (default: the end)")
    video.add_argument("--no-contacts", action="store_true", help="hide contact points and force vectors")
    video.add_argument("--no-overlay", action="store_true", help="no scene name / time caption")
    video.add_argument("-v", "--verbose", action="store_true", help="print the page's console")
    video.add_argument("--browser", default=None,
                       help="Chromium/Chrome executable (default: Playwright's Chromium, then Google Chrome)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.episode.is_file():
        parser.error(f"no such file: {args.episode}")

    episode = Episode(args.episode)
    roots = AssetRoots(args.repo.resolve() if args.repo else default_repo_root())
    if episode.scene_id not in scene_registry(roots):
        parser.error(f"scene {episode.scene_id!r} of this episode is not in {roots.repo}")
    if args.mp4 is not None:
        out = Path(args.mp4) if args.mp4 else _free_name(args.episode.with_suffix(".mp4"))
        try:
            width, height = (int(v) for v in args.size.lower().split("x"))
        except ValueError:
            parser.error("--size must look like 1920x1080")
        try:
            path = export_mp4(episode, roots, out, args, (width, height))
        finally:
            episode.close()
        print(f"saved {path}")
        return
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
