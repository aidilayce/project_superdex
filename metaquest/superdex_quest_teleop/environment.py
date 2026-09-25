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

"""Photoreal environment assets: a CC0 asset pack from Poly Haven.

SuperDex ships one Poly Haven HDRI (SuperDex Studio's lighting) but no room
environments. This module downloads, once, everything the client needs to
build textured 3D kitchens (see web/environments.js):

* HDRIs of real kitchens (lighting, reflections and optional 360 backdrop),
* PBR material sets (albedo, OpenGL normal, roughness) for the countertop,
  cabinets, backsplash, floor, walls, table top and stainless steel,
* glTF prop models (fruit, kitchen utensils, crockery, ...).

Everything is CC0 (https://polyhaven.com/license). Assets are chosen from
curated IDs first, then by keyword search over the live asset list, so the
pack keeps working when Poly Haven adds or renames assets. The pack is
cached under ~/.superdex_quest_teleop/assets with a manifest.json that the
server hands to the client.

    python -m superdex_quest_teleop.environment              # fetch the pack
    python -m superdex_quest_teleop.environment --resolution 2k
    python -m superdex_quest_teleop.environment --list-hdris
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger("superdex_quest_teleop")

API = "https://api.polyhaven.com"
CACHE_ROOT = Path.home() / ".superdex_quest_teleop"
CACHE_DIR = CACHE_ROOT / "environments"      # single HDRIs (fetch_kitchen_hdri)
PACK_DIR = CACHE_ROOT / "assets"             # the full pack
USER_AGENT = "superdex-quest-teleop/0.2 (+https://github.com/facebookresearch/project_superdex)"
FALLBACK_KEYWORDS = ("apartment", "living", "lounge", "home", "house", "interior", "room")

# Kitchen HDRIs (IDs verified on polyhaven.com), then keyword matches.
HDRIS: dict[str, list[str]] = {
    "kitchen_warm": ["kiara_interior"],      # cozy thatch kitchen-lounge
    "kitchen_morning": ["blinds"],           # morning kitchen, light through blinds
    "apartment": ["lebombo"],                # bright modern open-plan interior
}

# PBR material slots: curated IDs first, then keywords for the search.
MATERIALS: dict[str, tuple[list[str], list[str]]] = {
    "countertop": (["marble_01", "granite_tile", "marble_tiles"], ["marble", "granite"]),
    "cabinet": (["kitchen_wood", "oak_wood_planks", "wood_planks"], ["wood"]),
    "backsplash": (["long_white_tiles", "interior_tiles"], ["white tiles", "tiles"]),
    "floor": (["laminate_floor", "wood_floor", "laminate_floor_02"], ["laminate", "floor"]),
    "wall": (["white_plaster_02", "painted_plaster_wall", "white_stucco"], ["plaster"]),
    "tabletop": (["oak_wood_planks", "wood_floor", "kitchen_wood"], ["oak", "wood"]),
    "steel": (["metal_plate_02", "metal_plate"], ["brushed metal", "metal plate"]),
}
# Maps fetched per material: pack name -> Poly Haven map keys (first found).
MAPS = {
    "albedo": ("Diffuse", "diff", "Color", "albedo"),
    "normal": ("nor_gl", "Normal", "nor"),
    "roughness": ("Rough", "rough", "Roughness"),
}

# Props for the counters: curated model IDs, then models whose name or tags
# match these words in Poly Haven's food / kitchen categories.
PROPS = ["food_lime_01", "wooden_spoon"]
PROP_KEYWORDS = ("bowl", "plate", "mug", "cup", "jar", "bottle", "kettle", "pot", "pan",
                 "board", "spoon", "fruit", "lime", "lemon", "apple", "orange", "bread",
                 "towel", "soap", "sponge", "dish", "glass", "knife", "teapot")
MAX_PROPS = 10


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _download(url: str, target: Path, timeout: float) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(target)


def _words(asset_id: str, info: dict) -> str:
    return " ".join(
        [asset_id, str(info.get("name", ""))]
        + [str(t) for t in info.get("tags", [])]
        + [str(c) for c in info.get("categories", [])]
    ).lower().replace("_", " ")


def rank_hdris(assets: dict) -> list[tuple[int, str]]:
    """(score, id) for indoor HDRIs, best first: kitchens, then homes."""
    ranked = []
    for asset_id, info in assets.items():
        words = _words(asset_id, info)
        if "indoor" not in words:
            continue
        score = 100 if "kitchen" in words else 0
        score += sum(10 for k in FALLBACK_KEYWORDS if k in words)
        score += min(int(info.get("download_count", 0)) // 50000, 9)
        if score:
            ranked.append((score, asset_id))
    ranked.sort(reverse=True)
    return ranked


def _pick(assets: dict, curated: list[str], keywords: list[str] | tuple[str, ...]) -> str | None:
    for asset_id in curated:
        if asset_id in assets:
            return asset_id
    for keyword in keywords:
        for asset_id, info in sorted(assets.items()):
            if keyword in _words(asset_id, info):
                return asset_id
    return None


def _map_url(files: dict, keys: tuple[str, ...], resolution: str) -> str | None:
    lowered = {k.lower(): k for k in files}
    for key in keys:
        real = lowered.get(key.lower())
        if real is None:
            continue
        by_res = files[real]
        res = resolution if resolution in by_res else sorted(by_res)[0]
        formats = by_res[res]
        for fmt in ("jpg", "png"):
            if fmt in formats:
                return formats[fmt]["url"]
    return None


def fetch_pack(
    pack_dir: Path = PACK_DIR,
    resolution: str = "1k",
    hdri_resolution: str = "2k",
    api: str = API,
    timeout: float = 30.0,
) -> dict:
    """Download (or complete) the asset pack; returns the manifest dict.

    The manifest lists paths relative to ``pack_dir``:
    {"hdris": {name: path}, "materials": {slot: {"id", "albedo", "normal",
    "roughness"}}, "props": [{"id", "name", "path"}]}.
    """
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pack_dir / "manifest.json"
    manifest: dict = {"source": "https://polyhaven.com", "license": "CC0 1.0",
                      "hdris": {}, "materials": {}, "props": []}

    hdris = _get_json(f"{api}/assets?t=hdris", timeout)
    for name, curated in HDRIS.items():
        asset_id = _pick(hdris, curated, ["kitchen"] if "kitchen" in name else [])
        if asset_id is None:
            continue
        files = _get_json(f"{api}/files/{asset_id}", timeout)["hdri"]
        res = hdri_resolution if hdri_resolution in files else sorted(files)[0]
        rel = f"hdri/{asset_id}_{res}.hdr"
        _download(files[res]["hdr"]["url"], pack_dir / rel, timeout)
        manifest["hdris"][name] = {"id": asset_id, "path": rel}
        log.info("asset pack: HDRI %s -> %s", name, asset_id)

    textures = _get_json(f"{api}/assets?t=textures", timeout)
    for slot, (curated, keywords) in MATERIALS.items():
        asset_id = _pick(textures, curated, keywords)
        if asset_id is None:
            continue
        files = _get_json(f"{api}/files/{asset_id}", timeout)
        entry = {"id": asset_id}
        for name, keys in MAPS.items():
            url = _map_url(files, keys, resolution)
            if url:
                rel = f"textures/{asset_id}/{name}_{resolution}{Path(url).suffix}"
                _download(url, pack_dir / rel, timeout)
                entry[name] = rel
        if "albedo" in entry:
            manifest["materials"][slot] = entry
            log.info("asset pack: material %s -> %s", slot, asset_id)

    models = _get_json(f"{api}/assets?t=models", timeout)
    chosen = [m for m in PROPS if m in models]
    for asset_id, info in sorted(models.items()):
        if len(chosen) >= MAX_PROPS:
            break
        words = _words(asset_id, info)
        if asset_id not in chosen and ("food" in words or "kitchen" in words) and any(
            k in words for k in PROP_KEYWORDS
        ):
            chosen.append(asset_id)
    for asset_id in chosen[:MAX_PROPS]:
        files = _get_json(f"{api}/files/{asset_id}", timeout).get("gltf", {})
        if not files:
            continue
        res = resolution if resolution in files else sorted(files)[0]
        gltf = files[res]["gltf"]
        base = f"models/{asset_id}"
        main = f"{base}/{Path(gltf['url']).name}"
        _download(gltf["url"], pack_dir / main, timeout)
        for rel, item in (gltf.get("include") or {}).items():
            _download(item["url"], pack_dir / base / rel, timeout)
        manifest["props"].append({"id": asset_id, "name": models[asset_id].get("name", asset_id),
                                  "path": main})
        log.info("asset pack: prop %s", asset_id)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(pack_dir: Path = PACK_DIR) -> dict | None:
    path = pack_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def fetch_kitchen_hdri(
    cache_dir: Path = CACHE_DIR,
    asset_id: str | None = None,
    resolution: str = "2k",
    api: str = API,
    timeout: float = 20.0,
) -> Path:
    """Download (or reuse) a single kitchen HDRI; returns the local .hdr path."""
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
    _download(url, target, timeout)
    (cache_dir / f"{asset_id}.json").write_text(json.dumps({
        "source": f"https://polyhaven.com/a/{asset_id}", "license": "CC0 1.0", "file": url,
    }, indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the CC0 Poly Haven asset pack.")
    parser.add_argument("--resolution", default="1k", choices=("1k", "2k", "4k"),
                        help="texture/model resolution (1k is right for Quest)")
    parser.add_argument("--hdri-resolution", default="2k", choices=("1k", "2k", "4k", "8k"))
    parser.add_argument("--list-hdris", action="store_true", help="list indoor HDRIs and exit")
    parser.add_argument("--dir", type=Path, default=PACK_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.list_hdris:
        for score, asset_id in rank_hdris(_get_json(f"{API}/assets?t=hdris", 20.0))[:30]:
            print(f"{score:4d}  {asset_id}   https://polyhaven.com/a/{asset_id}")
        return
    manifest = fetch_pack(args.dir, args.resolution, args.hdri_resolution)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
