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

"""PC-side teleoperation server.

Serves the WebXR client to the Quest browser (and a desktop spectator
view), receives hand tracking over a WebSocket, runs SuperDex on a
dedicated physics thread and streams the simulated scene back.

Wire protocol (all WebSocket):

client -> server (JSON text)
  {"type": "hello", "role": "headset" | "viewer"}
  {"type": "hands", "head": [px,py,pz,qx,qy,qz,qw],
   "left":  {"tracked": bool, "p": [75 floats], "q": [100 floats], "r": [25 floats]},
   "right": {...}}          positions/rotations already in the physics frame
  {"type": "cmd", "cmd": "load_scene", "scene": "<id>"}
  {"type": "cmd", "cmd": "reset" | "next_scene" | "prev_scene" |
                         "record_start" | "record_stop" | "record_toggle"}

server -> client
  JSON  {"type": "hello", "scenes": [...], ...}
  JSON  {"type": "geometry", "scene": {...}, "actors": [...]}
  JSON  {"type": "status", ...}
  binary state frame: uint32 LE header length, UTF-8 JSON header, zero
        padding to a 4-byte boundary, then float32 payload:
        poses (N x 7) | deformable vertices (sum V_i x 3) |
        contact points (C x 3) | contact forces (C x 3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import ssl
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import hand_skeleton as hs
from .scenes import AssetRoots, SceneSpec, scene_registry
from .session import TeleopSession

log = logging.getLogger("superdex_quest_teleop")

WEB_DIR = Path(__file__).resolve().parent / "web"
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/"
INPUT_TIMEOUT = 0.5  # [s] without hand messages -> hands untracked
DISPLAY_CONTACTS = 256
DISPLAY_CONTACT_EVERY = 6  # steps, when not recording


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    # Plain HTTP: the desktop viewer and the Quest over USB (`adb reverse`
    # makes the PC reachable as http://localhost, which WebXR accepts).
    http_port: int = 8080
    # HTTPS with a self-signed certificate: the Quest over Wi-Fi. WebXR only
    # exists on secure pages, so plain-HTTP LAN requests are redirected here.
    https_port: int = 8443
    https: bool = True
    cert_dir: Path = Path.home() / ".superdex_quest_teleop"
    # Backdrop: an HDRI (.hdr), a 360 photo (.jpg/.png) or a model (.glb).
    # None with environment_auto fetches a CC0 kitchen HDRI from Poly Haven
    # in the background (cached); otherwise the procedural kitchen is used.
    environment: Path | None = None
    environment_auto: bool = True
    # Rigged display hands: a directory with left.glb / right.glb whose bones
    # carry the 25 WebXR joint names (default: WebXR generic-hand).
    hand_models: Path | None = None
    out_dir: Path = Path("recordings")
    scene: str = "box_and_blocks"
    contact_mode: str = "hand"
    min_contact_force: float = 0.0
    keep_self_contacts: bool = False
    hand_variant: str = "lowpoly"
    stream_hz: float = 60.0
    num_threads: int = -1
    synthetic: bool = False  # drive the right hand with a scripted grasp
    autostart_record: bool = False


# ----------------------------------------------------------------------------
# Physics thread
# ----------------------------------------------------------------------------


class PhysicsRunner(threading.Thread):
    """Owns every SuperDex object; runs the fixed-step loop in real time."""

    def __init__(self, config: ServerConfig, roots: AssetRoots, publish) -> None:
        super().__init__(name="superdex-physics", daemon=True)
        self.config = config
        self.roots = roots
        self.publish = publish  # (kind, payload) -> None, thread-safe
        self.registry: dict[str, SceneSpec] = scene_registry(roots)
        self.commands: queue.Queue = queue.Queue()
        self.session: TeleopSession | None = None
        self.running = True
        self.rtf = 1.0
        self._last_input_time = 0.0
        self._synthetic = None
        self._geometry_message: dict | None = None
        self.ready = threading.Event()

    # Thread-safe entry points ----------------------------------------------

    def submit(self, command: dict) -> None:
        self.commands.put(command)

    def set_hands(self, hands: dict[str, hs.HandFrame], head: np.ndarray | None) -> None:
        self._last_input_time = time.monotonic()
        session = self.session
        if session is not None and self._synthetic is None:
            session.set_input(hands, head)

    def scene_list(self) -> list[dict]:
        return [
            {"id": s.id, "name": s.name, "category": s.category,
             "description": s.description, "time_step": s.time_step}
            for s in self.registry.values()
        ]

    def geometry_message(self) -> dict | None:
        return self._geometry_message

    # Physics thread ----------------------------------------------------------

    def run(self) -> None:
        import superdex.physics as physics

        if not physics.is_initialized():
            physics.initialize(num_worker_threads=self.config.num_threads)
        physics.enable_file_cache(True)
        try:
            self._load(self.config.scene)
        except Exception:  # noqa: BLE001
            log.exception("could not load scene %s", self.config.scene)
            self._load(next(iter(self.registry)))
        if self.config.autostart_record:
            self._record_start()
        self.ready.set()

        next_time = time.monotonic()
        stream_period = 1.0 / self.config.stream_hz
        next_stream = next_time
        window_sim, window_start = 0.0, time.monotonic()
        steps_since_contacts = 0
        while self.running:
            self._drain_commands()
            session = self.session
            if session is None:
                time.sleep(0.01)
                continue
            dt = session.time_step
            now = time.monotonic()
            if next_time > now:
                time.sleep(next_time - now)
            elif now - next_time > 0.25:
                next_time = now  # fell behind: slow motion, never spiral
            next_time += dt

            if self._synthetic is not None:
                t = session.sim_time
                session.set_input({"right": self._synthetic.frame(t), "left": hs.HandFrame.untracked()})
            elif time.monotonic() - self._last_input_time > INPUT_TIMEOUT:
                session.set_input({side: hs.HandFrame.untracked() for side in session.hands})

            steps_since_contacts += 1
            want_contacts = steps_since_contacts >= DISPLAY_CONTACT_EVERY
            try:
                data = session.step(with_contacts=want_contacts)
            except Exception as exc:  # noqa: BLE001
                log.exception("physics step failed")
                self.publish("status", {"type": "status", "error": f"step failed: {exc}"})
                self._reset()
                continue
            if want_contacts or session.recorder is not None:
                steps_since_contacts = 0

            window_sim += dt
            elapsed = time.monotonic() - window_start
            if elapsed > 1.0:
                self.rtf = window_sim / elapsed
                window_sim, window_start = 0.0, time.monotonic()

            if time.monotonic() >= next_stream:
                next_stream = time.monotonic() + stream_period
                self.publish("frame", self._encode_frame(session, data))

        if self.session is not None:
            self._finish_recording()
            self.session.close()

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            name = cmd.get("cmd")
            try:
                if name == "load_scene":
                    self._load(cmd["scene"])
                elif name in ("next_scene", "prev_scene"):
                    ids = list(self.registry)
                    current = ids.index(self.session.spec.id) if self.session else 0
                    step = 1 if name == "next_scene" else -1
                    self._load(ids[(current + step) % len(ids)])
                elif name == "reset":
                    self._reset()
                elif name == "record_start":
                    self._record_start(cmd.get("metadata"))
                elif name == "record_stop":
                    self._finish_recording()
                elif name == "record_toggle":
                    if self.session and self.session.recorder:
                        self._finish_recording()
                    else:
                        self._record_start(cmd.get("metadata"))
                elif name == "stop":
                    self.running = False
                else:
                    log.warning("unknown command %r", name)
            except Exception as exc:  # noqa: BLE001
                log.exception("command %r failed", name)
                self.publish("status", {"type": "status", "error": f"{name} failed: {exc}"})

    def _load(self, scene_id: str) -> None:
        spec = self.registry[scene_id]
        self._finish_recording()
        if self.session is not None:
            self.session.close()
            self.session = None
        self.publish("status", {"type": "status", "message": f"loading {spec.name}..."})
        t0 = time.monotonic()
        session = TeleopSession(
            spec,
            self.roots,
            contact_mode=self.config.contact_mode,
            min_contact_force=self.config.min_contact_force,
            keep_self_contacts=self.config.keep_self_contacts,
            hand_variant=self.config.hand_variant,
            display_hand_models=self._display_models(),
        )
        self._synthetic = None
        if self.config.synthetic:
            from .synthetic import ScriptedGrasp

            tops = [r.actor.get_aabb_world().max[1] for r in session.object_records]
            self._synthetic = ScriptedGrasp(
                "right",
                object_top=float(min(max(tops), 0.2)) if tops else 0.05,
                kinematics=session.hands["right"].kinematics,
            )
        self._geometry_message = {
            "type": "geometry",
            "scene": {"id": spec.id, "name": spec.name, "description": spec.description,
                      "time_step": spec.time_step},
            "actors": [_jsonable_actor(a) for a in session.geometry()],
        }
        self.session = session
        log.info("loaded %s in %.1fs (%d actors)", spec.name, time.monotonic() - t0, len(session.actors))
        self.publish("geometry", self._geometry_message)
        self.publish("status", self.status())

    def _display_models(self) -> dict[str, Path] | None:
        folder = self.config.hand_models
        if folder is None:
            return None
        return {side: folder / f"{side}.glb" for side in hs.SIDES if (folder / f"{side}.glb").exists()}

    def _reset(self) -> None:
        if self.session is not None:
            self._load(self.session.spec.id)

    def _record_start(self, metadata: dict | None = None) -> None:
        if self.session is None:
            return
        meta = {"operator_input": "synthetic" if self._synthetic else "quest_webxr"}
        meta.update(metadata or {})
        path = self.session.start_recording(self.config.out_dir, meta)
        log.info("recording to %s", path)
        self.publish("status", self.status())

    def _finish_recording(self) -> None:
        if self.session is None:
            return
        result = self.session.stop_recording()
        if result is not None:
            path, steps = result
            log.info("saved %s (%d steps)", path, steps)
            status = self.status()
            status["saved"] = str(path)
            status["saved_steps"] = steps
            self.publish("status", status)

    def status(self) -> dict:
        s = self.session
        return {
            "type": "status",
            "scene": s.spec.id if s else None,
            "recording": bool(s and s.recorder),
            "file": str(s.recorder.path) if s and s.recorder else None,
            "contact_mode": self.config.contact_mode,
        }

    def _encode_frame(self, session: TeleopSession, data: dict) -> bytes:
        pose = data["pose"].astype(np.float32)
        verts = [d["surface_positions"].astype(np.float32) for d in data["deformables"]]
        contacts = session.last_contacts
        forces = contacts["force"]
        points = contacts["pos_owner"]
        if len(forces) > DISPLAY_CONTACTS:
            keep = np.argsort(-np.einsum("ij,ij->i", forces, forces))[:DISPLAY_CONTACTS]
            forces, points = forces[keep], points[keep]
        rec = session.recorder
        header = {
            "step": data["step"],
            "sim_time": round(data["sim_time"], 4),
            "rtf": round(self.rtf, 3),
            "recording": rec is not None,
            "recorded_steps": rec.num_steps if rec else 0,
            "num_actors": len(pose),
            "deformables": [[d.index, int(len(v))] for d, v in zip(session.deformables, verts)],
            "num_contacts": int(len(points)),
            "total_contacts": int(len(contacts["force"])),
            "tracked": {side: bool(h["tracked"]) for side, h in data["hands"].items()},
        }
        head_pose = data.get("head_pose")
        if head_pose is not None and np.all(np.isfinite(head_pose)):
            header["head"] = [round(float(v), 5) for v in head_pose]
        sides = [side for side in hs.SIDES if side in data["hands"]]
        header["hand_joints"] = sides
        head = json.dumps(header, separators=(",", ":")).encode()
        pad = (-(4 + len(head))) % 4
        payload = np.concatenate(
            [pose.reshape(-1)]
            + [v.reshape(-1) for v in verts]
            + [np.asarray(points, np.float32).reshape(-1), np.asarray(forces, np.float32).reshape(-1)]
            + [np.asarray(data["hands"][side]["display_joints"], np.float32).reshape(-1) for side in sides]
        ).astype("<f4")
        return struct.pack("<I", len(head)) + head + b"\0" * pad + payload.tobytes()


def _jsonable_actor(actor: dict) -> dict:
    out = {k: v for k, v in actor.items() if k not in ("positions", "indices")}
    if "positions" in actor:
        out["positions"] = np.round(actor["positions"].astype(np.float64), 5).tolist()
        out["indices"] = actor["indices"].astype(np.int64).tolist()
    return out


def parse_hands(message: dict) -> tuple[dict[str, hs.HandFrame], np.ndarray | None]:
    hands = {}
    for side in hs.SIDES:
        entry = message.get(side)
        if not entry or not entry.get("tracked") or len(entry.get("p", ())) != 3 * hs.NUM_JOINTS:
            hands[side] = hs.HandFrame.untracked()
            continue
        hands[side] = hs.HandFrame.from_flat(True, entry["p"], entry.get("q"), entry.get("r"))
    head = message.get("head")
    head_pose = np.asarray(head, dtype=np.float64) if head is not None and len(head) == 7 else None
    return hands, head_pose


# ----------------------------------------------------------------------------
# Web server
# ----------------------------------------------------------------------------


class TeleopServer:
    def __init__(self, config: ServerConfig, roots: AssetRoots | None = None) -> None:
        self.config = config
        self.roots = roots or AssetRoots()
        self.clients: dict[web.WebSocketResponse, asyncio.Queue] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner = PhysicsRunner(config, self.roots, self._publish_threadsafe)

    # Physics thread -> event loop.
    def _publish_threadsafe(self, kind: str, payload) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._broadcast, kind, payload)

    def _broadcast(self, kind: str, payload) -> None:
        for q in self.clients.values():
            if kind == "frame":
                # Frames are lossy: keep only the newest for slow clients.
                while q.qsize() >= 2:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            q.put_nowait((kind, payload))

    async def _index(self, request: web.Request) -> web.Response:
        host = (request.url.host or "").lower()
        local = host in ("localhost", "127.0.0.1", "::1", "[::1]")
        if request.secure is False and not local and self.config.https:
            # WebXR is unavailable on insecure pages: send LAN visitors to HTTPS.
            raise web.HTTPFound(f"https://{host}:{self.config.https_port}/")
        html = (WEB_DIR / "index.html").read_text()
        vendor = WEB_DIR / "vendor" / "three" / "build" / "three.module.js"
        base = "/vendor/three/" if vendor.exists() else THREE_CDN
        client = {
            "httpPort": self.config.http_port,
            "httpsPort": self.config.https_port if self.config.https else None,
            "environment": self._environment_info(),
            "handModels": self._hand_model_urls(),
            "ibl": "/ibl/studio_small_08_1k.hdr" if self._ibl_dir().exists() else None,
        }
        html = html.replace("{{THREE_BASE}}", base).replace("{{CLIENT_CONFIG}}", json.dumps(client))
        return web.Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-cache"})

    def _environment_info(self) -> dict | None:
        env = self.config.environment
        if env is None:
            return None
        suffix = env.suffix.lower()
        kind = "model" if suffix in (".glb", ".gltf") else "hdr" if suffix == ".hdr" else "panorama"
        return {"url": f"/environment/{env.name}", "kind": kind, "name": env.stem}

    def _ibl_dir(self) -> Path:
        return self.roots.repo / "superdex_studio" / "assets" / "ibl"

    def _hand_model_urls(self) -> dict[str, str]:
        if self.config.hand_models is not None:
            return {side: f"/hand_models/{side}.glb" for side in hs.SIDES
                    if (self.config.hand_models / f"{side}.glb").exists()}
        return {side: f"/vendor/webxr-input-profiles/generic-hand/{side}.glb" for side in hs.SIDES}

    def _fetch_environment(self) -> None:
        """Background: download a CC0 kitchen HDRI, then tell the clients."""
        from .environment import fetch_kitchen_hdri

        try:
            path = fetch_kitchen_hdri()
        except Exception as exc:  # noqa: BLE001 - offline is fine
            log.warning("no kitchen HDRI (%s); using the procedural kitchen. Fetch one later with "
                        "`python -m superdex_quest_teleop.environment` or pass --environment", exc)
            return
        self.config.environment = path
        log.info("environment: %s", path)
        self._publish_threadsafe("status", {"type": "environment", "environment": self._environment_info()})

    async def _environment(self, request: web.Request) -> web.StreamResponse:
        env = self.config.environment
        if env is None or request.match_info["name"] != env.name or not env.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(env)

    async def _scenes(self, request: web.Request) -> web.Response:
        return web.json_response(self.runner.scene_list())

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        self.clients[ws] = q
        sender = asyncio.create_task(self._sender(ws, q))
        await ws.send_json(
            {"type": "hello", "scenes": self.runner.scene_list(),
             "out_dir": str(Path(self.config.out_dir).resolve()),
             "synthetic": self.config.synthetic,
             "environment": self._environment_info()}
        )
        geometry = self.runner.geometry_message()
        if geometry is not None:
            await ws.send_json(geometry)
            await ws.send_json(self.runner.status())
        role = "viewer"
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    message = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                kind = message.get("type")
                if kind == "hands":
                    hands, head = parse_hands(message)
                    self.runner.set_hands(hands, head)
                elif kind == "cmd":
                    self.runner.submit(message)
                elif kind == "hello":
                    role = message.get("role", "viewer")
                    log.info("client connected as %s from %s", role, request.remote)
        finally:
            sender.cancel()
            self.clients.pop(ws, None)
            log.info("%s disconnected", role)
        return ws

    async def _sender(self, ws: web.WebSocketResponse, q: asyncio.Queue) -> None:
        try:
            while not ws.closed:
                kind, payload = await q.get()
                if kind == "frame":
                    await ws.send_bytes(payload)
                else:
                    await ws.send_json(payload)
        except (ConnectionResetError, asyncio.CancelledError):
            pass

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/api/scenes", self._scenes)
        app.router.add_get("/ws", self._ws)
        app.router.add_get("/environment/{name}", self._environment)
        app.router.add_static("/static/", WEB_DIR)
        if (WEB_DIR / "vendor").exists():
            app.router.add_static("/vendor/", WEB_DIR / "vendor")
        # Render meshes of the Meta XR hand and of the prefabs (read-only).
        hand_dir = self.roots.assets / "bots" / "hands" / "oculus_xr"
        if hand_dir.exists():
            app.router.add_static("/hand_assets/", hand_dir)
        if (self.roots.assets / "prefabs").exists():
            app.router.add_static("/prefab_assets/", self.roots.assets / "prefabs")
        if self._ibl_dir().exists():
            app.router.add_static("/ibl/", self._ibl_dir())
        if self.config.hand_models is not None:
            app.router.add_static("/hand_models/", self.config.hand_models)

        async def on_startup(app: web.Application) -> None:
            self.loop = asyncio.get_running_loop()
            self.runner.start()
            if self.config.environment is None and self.config.environment_auto:
                threading.Thread(target=self._fetch_environment, daemon=True).start()

        async def on_cleanup(app: web.Application) -> None:
            self.runner.submit({"cmd": "stop"})
            await asyncio.get_running_loop().run_in_executor(None, self.runner.join, 10.0)

        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        return app

    def ssl_context(self) -> ssl.SSLContext | None:
        if not self.config.https:
            return None
        cert, key = ensure_self_signed_cert(self.config.cert_dir)
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(cert, key)
        return ctx

    async def serve(self) -> None:
        """Serve HTTP (and HTTPS) until cancelled."""
        runner = web.AppRunner(self.make_app())
        await runner.setup()
        sites = [web.TCPSite(runner, self.config.host, self.config.http_port)]
        try:
            ctx = self.ssl_context()
        except Exception as exc:  # noqa: BLE001 - keep serving HTTP
            log.error("HTTPS disabled (%s); the Quest must connect over USB with adb reverse", exc)
            ctx = None
        if ctx is not None:
            sites.append(web.TCPSite(runner, self.config.host, self.config.https_port, ssl_context=ctx))
        try:
            for site in sites:
                await site.start()
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    def run(self) -> None:
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:
            pass


def ensure_self_signed_cert(cert_dir: Path, addresses: list[str] | None = None) -> tuple[Path, Path]:
    """A self-signed certificate for serving WebXR over the LAN (WebXR needs a
    secure context; the Quest browser lets you accept the warning once). It
    lists this machine's addresses so browsers only warn about the issuer."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    names = sorted(set(["127.0.0.1"] + (addresses if addresses is not None else lan_addresses())))
    stamp = cert_dir / "cert.names"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text() == ",".join(names):
        return cert, key
    san = ",".join(["DNS:localhost"] + [f"IP:{a}" for a in names])
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(cert), "-days", "825",
                "-subj", "/CN=superdex-quest-teleop", "-addext", f"subjectAltName={san}",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        _self_signed_with_cryptography(cert, key, names)
    stamp.write_text(",".join(names))
    return cert, key


def _self_signed_with_cryptography(cert: Path, key: Path, addresses: list[str]) -> None:
    """Fallback when the openssl CLI is missing (e.g. Windows)."""
    import datetime
    import ipaddress

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError(
            "HTTPS needs either the `openssl` command or `pip install cryptography`"
        ) from exc
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "superdex-quest-teleop")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(a)) for a in addresses]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .sign(private, hashes.SHA256())
    )
    key.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def lan_addresses() -> list[str]:
    """IPv4 addresses other devices on the LAN can reach this PC at."""
    import socket

    addresses = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addresses.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in addresses if not a.startswith("127."))
