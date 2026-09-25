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

"""Photographic kitchen backdrops: CC0 HDRIs from Poly Haven.

SuperDex itself ships one Poly Haven HDRI (SuperDex Studio's image-based
lighting) but no room environments. This module finds an indoor kitchen HDRI
through Poly Haven's public API, downloads it once and caches it; the client
uses it both as the 360-degree backdrop and as the lighting, so objects and
hands are lit by the same room they appear in.

    python -m superdex_quest_teleop.environment              # fetch a kitchen
    python -m superdex_quest_teleop.environment --list       # show candidates
    python -m superdex_quest_teleop.environment --id <asset> --resolution 4k
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("superdex_quest_teleop")

API = "https://api.polyhaven.com"
CACHE_DIR = Path.home() / ".superdex_quest_teleop" / "environments"
USER_AGENT = "superdex-quest-teleop/0.1 (+https://github.com/facebookresearch/project_superdex)"
# Used when no asset is tagged as a kitchen: residential interiors.
FALLBACK_KEYWORDS = ("apartment", "living", "lounge", "home", "house", "interior", "room")


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def rank_hdris(assets: dict) -> list[tuple[int, str]]:
    """(score, id) for indoor HDRIs, best first: kitchens, then homes."""
    ranked = []
    for asset_id, info in assets.items():
        words = " ".join(
            [asset_id, str(info.get("name", ""))]
            + [str(t) for t in info.get("tags", [])]
            + [str(c) for c in info.get("categories", [])]
        ).lower()
        if "indoor" not in words:
            continue
        score = 0
        if "kitchen" in words:
            score += 100
        score += sum(10 for k in FALLBACK_KEYWORDS if k in words)
        # Prefer popular, well-lit assets among equals.
        score += min(int(info.get("download_count", 0)) // 50000, 9)
        if score:
            ranked.append((score, asset_id))
    ranked.sort(reverse=True)
    return ranked


def fetch_kitchen_hdri(
    cache_dir: Path = CACHE_DIR,
    asset_id: str | None = None,
    resolution: str = "2k",
    api: str = API,
    timeout: float = 20.0,
) -> Path:
    """Download (or reuse) a kitchen HDRI; returns the local .hdr path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if asset_id is None:
        cached = sorted(cache_dir.glob(f"*_{resolution}.hdr"))
        if cached:
            return cached[0]
        ranked = rank_hdris(_get_json(f"{api}/assets?t=hdris", timeout))
        if not ranked:
            raise RuntimeError("Poly Haven returned no indoor HDRIs")
        asset_id = ranked[0][1]
    target = cache_dir / f"{asset_id}_{resolution}.hdr"
    if target.exists():
        return target
    files = _get_json(f"{api}/files/{asset_id}", timeout)
    url = files["hdri"][resolution]["hdr"]["url"]
    log.info("downloading %s (%s, CC0) from Poly Haven", asset_id, resolution)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(target)
    (cache_dir / f"{asset_id}.json").write_text(json.dumps({
        "source": f"https://polyhaven.com/a/{asset_id}", "license": "CC0 1.0", "file": url,
    }, indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a CC0 kitchen HDRI from Poly Haven.")
    parser.add_argument("--id", help="Poly Haven asset id (default: best kitchen match)")
    parser.add_argument("--resolution", default="2k", choices=("1k", "2k", "4k", "8k"))
    parser.add_argument("--list", action="store_true", help="list indoor candidates and exit")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.list:
        for score, asset_id in rank_hdris(_get_json(f"{API}/assets?t=hdris", 20.0))[:30]:
            print(f"{score:4d}  {asset_id}   https://polyhaven.com/a/{asset_id}")
        return
    print(fetch_kitchen_hdri(args.cache, args.id, args.resolution))


if __name__ == "__main__":
    main()
