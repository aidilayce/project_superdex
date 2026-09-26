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

"""``python -m superdex_quest_teleop``: run the Quest teleoperation server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .scenes import AssetRoots, default_repo_root, scene_registry
from .server import ServerConfig, TeleopServer, lan_addresses
from .session import CONTACT_MODES
from .workspace import available_environments



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m superdex_quest_teleop",
        description="Meta Quest hand-tracking teleoperation for SuperDex with "
        "per-point contact and force recording.",
    )
    parser.add_argument("--scene", default="kitchen_sponge", help="initial scene id (see --list-scenes)")
    parser.add_argument("--list-scenes", action="store_true", help="print the available scenes and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", "--port", type=int, default=8080,
                        help="plain HTTP port (desktop viewer, Quest over USB via adb reverse)")
    parser.add_argument("--https-port", type=int, default=8443,
                        help="HTTPS port with a self-signed certificate (Quest over Wi-Fi)")
    parser.add_argument("--no-https", action="store_true", help="serve plain HTTP only")
    parser.add_argument("--environment", default="kitchen_sink",
                        help="physical 3D environment: kitchen_sink (default: countertop, sink, "
                        "cabinets, floor), kitchen_island, studio, an imported room (rooms.py; the "
                        "presets retro_apartment, sherlock_221b and living_room import on first use "
                        "with SKETCHFAB_API_TOKEN set), all switchable with the Env button, or "
                        "table (a bare infinite tabletop); or a file shown as backdrop around the "
                        "island: an .hdr, an equirectangular 360 photo (.jpg/.png, e.g. of your own "
                        "kitchen) or a .glb/.gltf model")
    parser.add_argument("--no-download", action="store_true",
                        help="don't fetch the CC0 Poly Haven asset pack (textures, HDRIs, props)")
    parser.add_argument("--asset-dir", type=Path, default=None,
                        help="asset pack location (default: ~/.superdex_quest_teleop/assets)")
    parser.add_argument("--asset-resolution", default="1k", choices=("1k", "2k", "4k"),
                        help="texture resolution of the asset pack (1k suits Quest)")
    parser.add_argument("--hand-models", type=Path, default=None,
                        help="directory with left.glb/right.glb: rigged, textured hands whose bones "
                        "use the 25 WebXR joint names (default: WebXR generic-hand with skin shading)")
    parser.add_argument("--out", type=Path, default=Path("recordings"), help="episode output directory")
    parser.add_argument("--contacts", choices=CONTACT_MODES, default="hand",
                        help="hand: contacts involving the hands; all: also object/object and object/table")
    parser.add_argument("--min-contact-force", type=float, default=0.0,
                        help="drop contact points with |force| below this [N] (0 keeps near-contacts)")
    parser.add_argument("--keep-self-contacts", action="store_true",
                        help="also record contacts between links of the same hand")
    parser.add_argument("--hand-mesh", choices=("lowpoly", "highpoly"), default="lowpoly",
                        help="Meta XR hand collision mesh resolution")
    parser.add_argument("--threads", type=int, default=-1, help="SuperDex worker threads (-1 = auto)")
    parser.add_argument("--stream-hz", type=float, default=60.0, help="state streaming rate to clients")
    parser.add_argument("--time-budget", type=float, default=0.8,
                        help="cap each step's solve at this fraction of the time step so heavy "
                        "scenes (cloth) stay real time; 0 disables (exact but may run slower "
                        "than real time)")
    parser.add_argument("--hand-scale", type=float, default=1.0,
                        help="size of the simulated hands relative to the Meta XR hand asset "
                        "(normally set by the in-headset calibration when you enter VR)")
    parser.add_argument("--grip-strength", type=float, default=1.0,
                        help="finger torque cap multiplier (1 = 0.64 N m per joint); raise it if heavy "
                        "objects slip out of a firm grasp")
    parser.add_argument("--synthetic", action="store_true",
                        help="drive the right hand with a scripted grasp (no headset needed)")
    parser.add_argument("--record", action="store_true", help="start recording immediately")
    parser.add_argument("--repo", type=Path, default=None, help="SuperDex checkout (default: auto)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    roots = AssetRoots(args.repo.resolve() if args.repo else default_repo_root())
    registry = scene_registry(roots)
    if args.list_scenes:
        for spec in registry.values():
            print(f"{spec.id:28s} {spec.category:10s} {spec.name}: {spec.description}")
        return
    if args.scene not in registry:
        parser.error(f"unknown scene {args.scene!r}; use --list-scenes")

    environment = None
    scene_environment = args.environment
    from .rooms import PRESETS

    if args.environment in PRESETS and args.environment not in available_environments():
        # A preset room that isn't imported yet: import it now (Sketchfab token needed).
        import os

        from .rooms import import_room

        print(f"importing the {args.environment} room (one-time)...")
        import_room(args.environment, args.environment, os.environ.get("SKETCHFAB_API_TOKEN"))
    if args.environment not in available_environments() + ["table"]:
        environment = Path(args.environment).expanduser().resolve()
        if not environment.is_file():
            parser.error(f"unknown environment {args.environment!r}: use one of "
                         f"{', '.join(available_environments())} or an existing file "
                         "(import rooms with `python -m superdex_quest_teleop.rooms add`)")
        scene_environment = "kitchen_island"
    if args.hand_models is not None and not (args.hand_models / "right.glb").exists():
        parser.error(f"{args.hand_models} must contain left.glb and right.glb")
    config = ServerConfig(
        host=args.host,
        http_port=args.http_port,
        https_port=args.https_port,
        https=not args.no_https,
        environment=environment,
        scene_environment=scene_environment,
        environment_auto=not args.no_download,
        asset_resolution=args.asset_resolution,
        **({"pack_dir": args.asset_dir.expanduser().resolve()} if args.asset_dir else {}),
        hand_models=args.hand_models.resolve() if args.hand_models else None,
        out_dir=args.out,
        scene=args.scene,
        contact_mode=args.contacts,
        min_contact_force=args.min_contact_force,
        keep_self_contacts=args.keep_self_contacts,
        hand_variant=args.hand_mesh,
        stream_hz=args.stream_hz,
        num_threads=args.threads,
        time_budget=args.time_budget if args.time_budget > 0 else None,
        hand_scale=args.hand_scale,
        grip_strength=args.grip_strength,
        synthetic=args.synthetic,
        autostart_record=args.record,
    )
    print("SuperDex Quest teleop server")
    print(f"  desktop viewer : http://localhost:{args.http_port}/")
    print(f"  Quest over USB : adb reverse tcp:{args.http_port} tcp:{args.http_port}, then open "
          f"http://localhost:{args.http_port}/ in the Quest browser")
    if not args.no_https:
        for address in lan_addresses():
            print(f"  Quest over LAN : https://{address}:{args.https_port}/   "
                  "(accept the self-signed certificate once: Advanced -> Proceed)")
    print(f"  recordings     : {args.out.resolve()}")
    TeleopServer(config, roots).run()


if __name__ == "__main__":
    main()
