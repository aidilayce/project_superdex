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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m superdex_quest_teleop",
        description="Meta Quest hand-tracking teleoperation for SuperDex with "
        "per-point contact and force recording.",
    )
    parser.add_argument("--scene", default="box_and_blocks", help="initial scene id (see --list-scenes)")
    parser.add_argument("--list-scenes", action="store_true", help="print the available scenes and exit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http-port", "--port", type=int, default=8080,
                        help="plain HTTP port (desktop viewer, Quest over USB via adb reverse)")
    parser.add_argument("--https-port", type=int, default=8443,
                        help="HTTPS port with a self-signed certificate (Quest over Wi-Fi)")
    parser.add_argument("--no-https", action="store_true", help="serve plain HTTP only")
    parser.add_argument("--environment", type=Path, default=None,
                        help="backdrop instead of the built-in kitchen: a .glb/.gltf model or an "
                        "equirectangular 360 photo (.jpg/.png), e.g. of your own kitchen")
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

    if args.environment is not None and not args.environment.is_file():
        parser.error(f"environment file not found: {args.environment}")
    config = ServerConfig(
        host=args.host,
        http_port=args.http_port,
        https_port=args.https_port,
        https=not args.no_https,
        environment=args.environment.resolve() if args.environment else None,
        out_dir=args.out,
        scene=args.scene,
        contact_mode=args.contacts,
        min_contact_force=args.min_contact_force,
        keep_self_contacts=args.keep_self_contacts,
        hand_variant=args.hand_mesh,
        stream_hz=args.stream_hz,
        num_threads=args.threads,
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
