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

"""Download three.js from the npm registry into the web client's vendor dir.

By default the client loads three.js from cdn.jsdelivr.net, which needs the
headset to reach the internet. Run this once to serve it from the PC instead
(offline labs, locked-down networks):

    python metaquest/tools/vendor_three.py
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

VERSION = "0.160.0"
FILES = ("build/three.module.js", "examples/jsm/controls/OrbitControls.js", "LICENSE")
DEST = Path(__file__).resolve().parents[1] / "superdex_quest_teleop" / "web" / "vendor" / "three"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--tarball", type=Path, help="use a local `npm pack three` tarball")
    args = parser.parse_args()
    if args.tarball:
        data = args.tarball.read_bytes()
    else:
        url = f"https://registry.npmjs.org/three/-/three-{args.version}.tgz"
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response:
            data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name in FILES:
            member = tar.extractfile(f"package/{name}")
            if member is None:
                raise SystemExit(f"{name} missing from the three.js package")
            out = DEST / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(member.read())
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
