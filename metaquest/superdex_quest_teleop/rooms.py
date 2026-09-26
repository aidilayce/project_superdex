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

"""Rooms from 3D models (Sketchfab or any glTF/GLB) as physical environments.

    python -m superdex_quest_teleop.rooms add retro_apartment        # a preset
    python -m superdex_quest_teleop.rooms add https://sketchfab.com/3d-models/<name>-<uid> --name my_room
    python -m superdex_quest_teleop.rooms add ~/Downloads/room.zip --name my_room   # .zip/.gltf/.glb
    python -m superdex_quest_teleop.rooms surfaces my_room           # list work-surface candidates
    python -m superdex_quest_teleop.rooms add my_room --surface 2    # re-import on another surface
    python -m superdex_quest_teleop.rooms list

Sketchfab downloads need a (free) API token: sketchfab.com -> Settings ->
Password & API -> API token; pass ``--token`` or set SKETCHFAB_API_TOKEN.
Each model's license and author are recorded and shown in the viewer
(CC Attribution models require the credit).

The import makes the model physical around a work surface:

1. **Units and floor.** glTF is Y-up. The floor is the lowest large
   upward-facing area, the ceiling the largest downward-facing one above
   it; the unit scale (m, cm, mm, inch, ft) is the one that makes the room
   2.2-4.5 m high, or its floor plan 2-20 m across when it has no ceiling.
2. **Work surface.** Upward-facing areas 0.35-1.15 m above the floor (tables,
   desks, counters) of at least 45 x 30 cm, preferring 0.7-0.95 m (standing
   work height) and larger areas. The physics origin goes on its center,
   its long side along X, the operator on its freer side (+Z).
3. **Colliders.** Every model part within 1.4 m of the surface (split into
   connected pieces) becomes static convex colliders: a CoACD convex
   decomposition for pieces near the surface, convex hulls further away.
   Room shells (floor, walls and ceiling as one piece around the table) are
   left out: an analytic floor plane replaces the floor. The work surface
   itself is an exact box, flat at y = 0.

In VR the room keeps its real size: the surface stays at its modeled height
above your real floor (instead of 0.5 m below your eyes as in the kitchens).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("superdex_quest_teleop")

ROOMS_DIR = Path(os.environ.get("SUPERDEX_TELEOP_ROOMS", Path.home() / ".superdex_quest_teleop" / "rooms"))
SKETCHFAB_API = "https://api.sketchfab.com/v3/models"

# Downloadable ("Download Free 3D model") Sketchfab rooms, by uid. The rooms
# first asked for (Archilogic's "221B Baker Street", "Modern Retro
# Apartment", "Living Room" 08c0ae41...) have downloads disabled by their
# authors, so these are the closest downloadable alternatives; find more with
# `rooms search`.
PRESETS = {
    "sherlock_221b": {
        "uid": "d72ffc510efe48d48bf519612bbaac42",
        "title": "Isometric Room - 221B Baker Street (cjbarron)",
        "url": "https://sketchfab.com/3d-models/isometric-room-221b-baker-street-d72ffc510efe48d48bf519612bbaac42",
    },
    "victorian_living_room": {
        "uid": "31855a7fc439491c97d621eb25c59bcc",
        "title": "Victorian Living Room (justynkrupa15)",
        "url": "https://sketchfab.com/3d-models/victorian-living-room-31855a7fc439491c97d621eb25c59bcc",
    },
    "retro_apartment": {
        "uid": "400c9069181a4342a7142433dfa3466e",
        "title": "Modern apartment interior, old style (Katydid)",
        "url": "https://sketchfab.com/3d-models/modern-apartment-interior-400c9069181a4342a7142433dfa3466e",
    },
    "apartment": {
        "uid": "1f9a624a807f457488b9ae8e101d76d0",
        "title": "Living room + kitchen + bedroom (Katydid)",
        "url": "https://sketchfab.com/3d-models/living-roomkitchenbedroom-1f9a624a807f457488b9ae8e101d76d0",
    },
    "living_room": {
        "uid": "ec7179648b1e43739104994d64e95673",
        "title": "Modern Living Room (Visthetique)",
        "url": "https://sketchfab.com/3d-models/modern-living-room-ec7179648b1e43739104994d64e95673",
    },
    "cozy_living_room": {
        "uid": "581238dc5fda4dc990571cdc02827783",
        "title": "Cozy living room baked (ChristyHsu)",
        "url": "https://sketchfab.com/3d-models/cozy-living-room-baked-581238dc5fda4dc990571cdc02827783",
    },
}

UNIT_SCALES = (1.0, 0.01, 0.001, 0.0254, 0.3048, 0.1)
SURFACE_HEIGHT = (0.35, 1.15)  # [m] above the floor
SURFACE_MIN = (0.45, 0.30)  # [m] length x width
REACH = 1.4  # [m] colliders around the surface center
COACD_REACH = 0.9  # [m] exact (decomposed) colliders this close, hulls beyond
MAX_PARTS = 400


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------


def sketchfab_uid(text: str) -> str | None:
    if text in PRESETS:
        return PRESETS[text]["uid"]
    m = re.search(r"([0-9a-f]{32})", text)
    return m.group(1) if m else None


def _http_json(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def sketchfab_download(uid: str, dest: Path, token: str | None) -> tuple[Path, dict]:
    """Download a model's glTF archive (needs an API token). Returns the
    extracted folder and the model's metadata (name, author, license)."""
    meta = _http_json(f"{SKETCHFAB_API}/{uid}")
    info = {
        "title": meta.get("name"),
        "author": (meta.get("user") or {}).get("displayName") or (meta.get("user") or {}).get("username"),
        "author_url": (meta.get("user") or {}).get("profileUrl"),
        "license": (meta.get("license") or {}).get("label"),
        "license_url": (meta.get("license") or {}).get("url"),
        "source": meta.get("viewerUrl") or f"https://sketchfab.com/3d-models/{uid}",
        "downloadable": meta.get("isDownloadable"),
    }
    if not meta.get("isDownloadable"):
        raise SystemExit(
            f"{info['title']!r} ({info['source']}) is not downloadable: its author disabled downloads on "
            "Sketchfab, so no token can fetch it. Find downloadable rooms with "
            "`rooms search \"living room\"` (or pick a preset: " + ", ".join(PRESETS) + ").")
    if not token:
        raise SystemExit(
            "Sketchfab downloads need your API token: sketchfab.com -> Settings -> Password & API -> "
            "API token. Pass --token TOKEN or set SKETCHFAB_API_TOKEN. (Or download the glTF from the "
            "model page yourself and run `rooms add path/to/model.zip --name NAME`.)"
        )
    links = _http_json(f"{SKETCHFAB_API}/{uid}/download", token)
    choice = links.get("glb") or links.get("gltf")
    if not choice:
        raise SystemExit(f"no glTF download offered for {uid}: {sorted(links)}")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ("model.glb" if "glb" in links else "model.zip")
    log.info("downloading %s (%.1f MB)", info["title"], (choice.get("size") or 0) / 1e6)
    with urllib.request.urlopen(choice["url"], timeout=600) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f)
    return archive, info


def sketchfab_search(query: str, count: int = 24, token: str | None = None) -> list[dict]:
    """Downloadable Sketchfab models matching ``query`` (most liked first)."""
    import urllib.parse

    params = urllib.parse.urlencode({"type": "models", "q": query, "downloadable": "true",
                                     "count": count, "sort_by": "-likeCount"})
    data = _http_json(f"https://api.sketchfab.com/v3/search?{params}", token)
    out = []
    for m in data.get("results", []):
        lic = m.get("license")
        out.append({
            "uid": m.get("uid"), "name": m.get("name"),
            "author": (m.get("user") or {}).get("displayName") or (m.get("user") or {}).get("username"),
            "license": lic.get("label") if isinstance(lic, dict) else lic,
            "faces": m.get("faceCount"), "likes": m.get("likeCount"),
            "url": m.get("viewerUrl") or f"https://sketchfab.com/3d-models/{m.get('uid')}",
        })
    return out


def _extract(path: Path, dest: Path) -> Path:
    """The model file (.glb/.gltf) of a download or local path."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        path = dest
    if path.is_dir():
        found = sorted(path.rglob("*.glb")) + sorted(path.rglob("*.gltf"))
        if not found:
            raise SystemExit(f"no .glb/.gltf in {path}")
        return found[0]
    return path


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def load_model(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(node name, world vertices, faces) of every mesh in a glTF scene."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        v = np.asarray(geom.vertices, np.float64) @ transform[:3, :3].T + transform[:3, 3]
        parts.append((str(node), v, np.asarray(geom.faces, np.int64)))
    if not parts:
        raise SystemExit(f"{path}: no triangle meshes")
    return parts


def _triangles(parts):
    tris = np.concatenate([v[f] for _, v, f in parts])
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(2 * area, 1e-18)[:, None]
    owner = np.concatenate([np.full(len(f), i) for i, (_, _, f) in enumerate(parts)])
    return tris, area, normal, owner


def _horizontal_area_by_height(tris, area, normal, up: bool, bin_size: float):
    sel = normal[:, 1] > 0.95 if up else normal[:, 1] < -0.95
    y = tris[sel, :, 1].mean(1)
    if not len(y):
        return np.zeros(0), np.zeros(0)
    edges = np.arange(y.min() - bin_size, y.max() + 2 * bin_size, bin_size)
    hist, _ = np.histogram(y, bins=edges, weights=area[sel])
    return 0.5 * (edges[1:] + edges[:-1]), hist


def detect_scale_and_floor(tris, area, normal) -> tuple[float, float, float | None]:
    """(unit scale, floor y, room height or None) in model units -> meters."""
    extent = tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)
    bin_size = max(extent[1] / 400.0, 1e-6)
    ys, up = _horizontal_area_by_height(tris, area, normal, True, bin_size)
    if not len(ys):
        raise SystemExit("no upward-facing surfaces: is the model Y-up?")
    floor_y = float(ys[np.argmax(up >= 0.3 * up.max())])
    yd, down = _horizontal_area_by_height(tris, area, normal, False, bin_size)
    height = None
    if len(yd):
        above = yd > floor_y + 0.2 * extent[1]
        if above.any() and down[above].max() > 0.2 * up.max():
            height = float(yd[above][np.argmax(down[above])] - floor_y)
    candidates = []
    for s in UNIT_SCALES:
        if height is not None:
            h = height * s
            if 2.2 <= h <= 4.5:
                candidates.append((abs(h - 2.7), s))
        else:
            span = max(extent[0], extent[2]) * s
            if 2.0 <= span <= 20.0:
                candidates.append((abs(span - 6.0), s))
    scale = min(candidates)[1] if candidates else 1.0
    return scale, floor_y, height


@dataclass
class Surface:
    height: float  # [m] above the floor
    center: np.ndarray  # (x, z) in meters (model axes, scaled)
    axis: np.ndarray  # unit (x, z) of the long side
    size: tuple[float, float]  # long, short [m]
    area: float
    node: str
    score: float

    def describe(self) -> str:
        return (f"{self.height:.2f} m high, {self.size[0]:.2f} x {self.size[1]:.2f} m, "
                f"{self.area:.2f} m2 ({self.node})")


def _rasterize(grid: np.ndarray, tris: np.ndarray) -> None:
    """Mark the grid cells whose centers lie in (or next to) 2-D triangles
    given in cell units."""
    for t in tris:
        lo = np.floor(t.min(0)).astype(int)
        hi = np.ceil(t.max(0)).astype(int)
        grid[t[:, 0].astype(int), t[:, 1].astype(int)] = True
        if (hi - lo).prod() <= 1:
            continue
        xs, zs = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        p = np.stack([xs.ravel() + 0.5, zs.ravel() + 0.5], 1)
        a, b, c = t
        v0, v1, v2 = c - a, b - a, p - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-12:
            continue
        d20, d21 = v2 @ v0, v2 @ v1
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        inside = (u >= -0.05) & (v >= -0.05) & (u + v <= 1.05)
        q = p[inside].astype(int)
        q = q[(q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < grid.shape[0]) & (q[:, 1] < grid.shape[1])]
        grid[q[:, 0], q[:, 1]] = True


def find_surfaces(tris, area, normal, owner, parts, floor_y: float) -> list[Surface]:
    """Horizontal, upward-facing, table-height areas large enough to work on."""
    from scipy import ndimage

    up = normal[:, 1] > 0.97
    y = tris[:, :, 1].mean(1) - floor_y
    sel = up & (y > SURFACE_HEIGHT[0]) & (y < SURFACE_HEIGHT[1])
    out: list[Surface] = []
    if not sel.any():
        return out
    cell = 0.02
    heights = np.round(y[sel] / 0.01) * 0.01
    for h in np.unique(heights):
        mask = sel.copy()
        mask[sel] = np.abs(heights - h) < 1e-6
        if area[mask].sum() < 0.1:
            continue
        pts = tris[mask][:, :, [0, 2]]
        lo = pts.reshape(-1, 2).min(0)
        dims = np.ceil((pts.reshape(-1, 2).max(0) - lo) / cell).astype(int) + 2
        grid = np.zeros(dims, bool)
        _rasterize(grid, (pts - lo) / cell)
        grid = ndimage.binary_closing(grid, iterations=2)
        labels, count = ndimage.label(grid)
        for k in range(1, count + 1):
            cells = np.argwhere(labels == k)
            a = len(cells) * cell * cell
            if a < 0.12:
                continue
            xy = lo + (cells + 0.5) * cell
            center = xy.mean(0)
            cov = np.cov((xy - center).T)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, 1]
            proj = (xy - center) @ np.stack([axis, evecs[:, 0]], 1)
            size = proj.max(0) - proj.min(0) + cell
            center = center + np.stack([axis, evecs[:, 0]], 1) @ (0.5 * (proj.max(0) + proj.min(0)))
            if size[0] < SURFACE_MIN[0] or size[1] < SURFACE_MIN[1]:
                continue
            pref = 1.0 if 0.7 <= h <= 0.95 else 0.7 if 0.6 <= h <= 1.05 else 0.35
            fill = a / (size[0] * size[1])
            node_counts = np.bincount(owner[mask], minlength=len(parts))
            out.append(Surface(float(h), center, axis, (float(size[0]), float(size[1])), a,
                               parts[int(np.argmax(node_counts))][0], pref * min(a, 1.5) * min(1.0, fill + 0.3)))
    out.sort(key=lambda s: -s.score)
    return out


def _free_side(tris, surface: Surface, floor_y: float) -> float:
    """+1 or -1: the side (along the short axis) with more free floor space."""
    side = np.array([-surface.axis[1], surface.axis[0]])
    c = tris.reshape(-1, 3)
    body = (c[:, 1] > floor_y + 0.15) & (c[:, 1] < floor_y + 1.7)
    xz = c[body][:, [0, 2]]
    scores = []
    for sign in (1.0, -1.0):
        probe = surface.center + sign * side * (0.5 * surface.size[1] + 0.45)
        scores.append(np.sum(np.linalg.norm(xz - probe, axis=1) < 0.35))
    return 1.0 if scores[0] <= scores[1] else -1.0


def physics_from_model(scale: float, floor_y: float, surface: Surface, facing: float) -> np.ndarray:
    """4x4: model units -> physics frame (surface center at the origin, top
    at y = 0, long side along X, the operator's free side at +Z)."""
    side = np.array([-surface.axis[1], surface.axis[0]]) * facing  # (x, z) toward the operator
    # Right-handed with Y up: X = Y x Z, i.e. X = (s_z, -s_x) for Z = (s_x, s_z).
    x_axis = np.array([side[1], 0.0, -side[0]])
    z_axis = np.array([side[0], 0.0, side[1]])
    R = np.stack([x_axis, [0.0, 1.0, 0.0], z_axis])  # rows: physics axes in model coordinates
    T = np.eye(4)
    T[:3, :3] = R * scale
    top = np.array([surface.center[0], floor_y * scale + surface.height, surface.center[1]])
    T[:3, 3] = -(R @ top)
    return T


# ----------------------------------------------------------------------------
# Colliders
# ----------------------------------------------------------------------------


def build_colliders(parts, T: np.ndarray, floor: float, surface: Surface, use_coacd: bool = True):
    """Convex pieces (physics frame) for the model parts around the surface."""
    import trimesh

    pieces = []
    shells = []
    for name, v, f in parts:
        w = v @ T[:3, :3].T + T[:3, 3]
        mesh = trimesh.Trimesh(w, f, process=True)
        for comp in mesh.split(only_watertight=False):
            lo, hi = comp.bounds
            # Outside reach, below the floor, or a room shell around the table.
            if np.any(hi[[0, 2]] < -REACH) or np.any(lo[[0, 2]] > REACH) or hi[1] < floor + 0.01:
                continue
            if lo[1] > floor + 2.2:
                continue
            spans = hi - lo
            if (lo[0] < 0 < hi[0] and lo[2] < 0 < hi[2] and max(spans[0], spans[2]) > 3.0):
                shells.append(comp)  # floor/walls/ceiling around the table: walls below
                continue
            if max(spans) < 0.01 or len(comp.faces) < 4:
                continue
            pieces.append((name, comp))
    parts_out = []
    for name, comp in pieces:
        center = comp.bounds.mean(0)
        near = np.linalg.norm(center[[0, 2]]) < COACD_REACH
        hull = comp.convex_hull
        concave = comp.is_watertight and comp.volume < 0.7 * hull.volume or not comp.is_watertight
        if use_coacd and near and concave and max(comp.extents) > 0.08:
            try:
                import coacd

                coacd.set_log_level("error")
                result = coacd.run_coacd(coacd.Mesh(comp.vertices, comp.faces), threshold=0.08,
                                         max_convex_hull=16, preprocess_resolution=40)
                for vv, ff in result:
                    parts_out.append((name, trimesh.Trimesh(vv, ff).convex_hull))
                continue
            except Exception as exc:  # noqa: BLE001 - coacd missing or failed
                log.debug("coacd failed on %s: %s", name, exc)
        if hull.volume > 1e-7:
            parts_out.append((name, hull))
    for wall in _walls(shells, floor):
        parts_out.append(("wall", wall))
    # The work surface itself: an exact slab, flat at y = 0.
    L, W = surface.size
    slab = trimesh.creation.box(extents=(L - 0.01, 0.03, W - 0.01))
    slab.apply_translation((0, -0.015, 0))
    parts_out.sort(key=lambda p: np.linalg.norm(p[1].bounds.mean(0)[[0, 2]]))
    if len(parts_out) > MAX_PARTS:
        log.warning("%d collider pieces, keeping the %d nearest", len(parts_out), MAX_PARTS)
        parts_out = parts_out[:MAX_PARTS]
    return parts_out, slab


def _walls(shells, floor: float, thickness: float = 0.1):
    """Box slabs behind the large vertical faces of room shells within reach
    (the wall behind a desk), clipped to the reach region."""
    import trimesh

    out = []
    for shell in shells:
        n = shell.face_normals
        vertical = np.abs(n[:, 1]) < 0.1
        if not vertical.any():
            continue
        centers = shell.triangles_center[vertical]
        normals = n[vertical]
        areas = shell.area_faces[vertical]
        # Group faces by plane: direction (1 degree) and offset (1 cm).
        heading = np.round(np.degrees(np.arctan2(normals[:, 2], normals[:, 0]))).astype(int)
        offset = np.round(np.einsum("ij,ij->i", centers, normals) / 0.01).astype(int)
        for key in set(zip(heading, offset)):
            sel = (heading == key[0]) & (offset == key[1])
            if areas[sel].sum() < 0.5:
                continue
            normal = normals[sel].mean(0)
            normal /= np.linalg.norm(normal)
            d = float(np.dot(centers[sel].mean(0), normal))
            if abs(d) > REACH:  # plane too far from the work surface
                continue
            tangent = np.array([-normal[2], 0.0, normal[0]])
            pts = shell.triangles[vertical][sel].reshape(-1, 3)
            t = pts @ tangent
            lo_t, hi_t = max(t.min(), -REACH), min(t.max(), REACH)
            lo_y, hi_y = max(pts[:, 1].min(), floor), min(pts[:, 1].max(), floor + 2.5)
            if hi_t - lo_t < 0.2 or hi_y - lo_y < 0.2:
                continue
            box = trimesh.creation.box(extents=(hi_t - lo_t, hi_y - lo_y, thickness))
            # Box local Z along the face normal; the slab sits behind the face.
            R = np.eye(4)
            R[:3, 0] = tangent
            R[:3, 1] = [0.0, 1.0, 0.0]
            R[:3, 2] = normal
            R[:3, 3] = normal * (d - thickness / 2) + tangent * (lo_t + hi_t) / 2 + [0, (lo_y + hi_y) / 2, 0] \
                - normal * 0  # (the face plane passes through normal * d)
            # Remove the tangent/normal components of the vertical offset.
            box.apply_transform(R)
            out.append(box)
    return out


# ----------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------


def import_room(name: str, source: str | Path, token: str | None = None, surface_index: int = 0,
                scale: float | None = None, use_coacd: bool = True, root: Path | None = None) -> Path:
    import trimesh

    room_dir = Path(root or ROOMS_DIR) / name
    src_dir = room_dir / "source"
    info_path = room_dir / "room.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    source = str(source)
    info = {k: old.get(k) for k in ("title", "author", "author_url", "license", "license_url", "source")}
    uid = sketchfab_uid(source) if not Path(source).expanduser().exists() else None
    if uid and not (src_dir.exists() and old.get("uid") == uid):
        if src_dir.exists():
            shutil.rmtree(src_dir)
        archive, meta = sketchfab_download(uid, src_dir, token)
        info.update(meta)
        model = _extract(archive, src_dir)
    elif uid:
        model = room_dir / old["model"]
    elif old.get("model") and source == name:
        model = room_dir / old["model"]
    else:
        path = Path(source).expanduser()
        if src_dir.exists():
            shutil.rmtree(src_dir)
        src_dir.mkdir(parents=True)
        if path.is_dir():
            shutil.copytree(path, src_dir, dirs_exist_ok=True)
            model = _extract(src_dir, src_dir)
        elif path.suffix.lower() == ".zip":
            model = _extract(path, src_dir)
        else:
            # A .gltf references buffers/textures next to it: copy the folder.
            if path.suffix.lower() == ".gltf":
                shutil.copytree(path.parent, src_dir, dirs_exist_ok=True)
                model = src_dir / path.name
            else:
                model = Path(shutil.copy2(path, src_dir / path.name))
        info.setdefault("source", str(path))
        info["title"] = info.get("title") or path.stem
    if name in PRESETS:
        info["title"] = info.get("title") or PRESETS[name]["title"]
        info["source"] = info.get("source") or PRESETS[name]["url"]

    parts = load_model(model)
    tris, area, normal, owner = _triangles(parts)
    auto_scale, floor_raw, height_raw = detect_scale_and_floor(tris, area, normal)
    s = scale or auto_scale
    # Work in meters for the surface search.
    tris_m = tris * s
    surfaces = find_surfaces(tris_m, area * s * s, normal, owner, parts, floor_raw * s)
    if not surfaces:
        raise SystemExit("no work surface 0.35-1.15 m high and at least 45 x 30 cm found; "
                         "check --scale (detected %g)" % s)
    if not 0 <= surface_index < len(surfaces):
        raise SystemExit(f"--surface must be 0..{len(surfaces) - 1}")
    surface = surfaces[surface_index]
    facing = _free_side(tris_m, surface, floor_raw * s)
    T = physics_from_model(s, floor_raw, surface, facing)
    floor = -surface.height
    colliders, slab = build_colliders(parts, T, floor, surface, use_coacd)

    room_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        room_dir / "colliders.npz",
        **{f"v{i}": np.asarray(m.vertices, np.float32) for i, (_, m) in enumerate(colliders)},
        **{f"f{i}": np.asarray(m.faces, np.int32) for i, (_, m) in enumerate(colliders)},
        slab_v=np.asarray(slab.vertices, np.float32), slab_f=np.asarray(slab.faces, np.int32),
    )
    overlay = trimesh.Scene([m for _, m in colliders] + [slab])
    overlay.export(str(room_dir / "colliders.glb"), file_type="glb")
    info.update({
        "name": name, "uid": uid, "model": model.relative_to(room_dir).as_posix(),
        "scale": s, "scale_detected": auto_scale, "room_height_m": height_raw * s if height_raw else None,
        "physics_from_model": T.tolist(), "counter_height": surface.height,
        "surface": {"index": surface_index, "size": list(surface.size), "area": surface.area,
                    "node": surface.node, "description": surface.describe()},
        "surfaces": [x.describe() for x in surfaces[:12]],
        "num_colliders": len(colliders),
    })
    info_path.write_text(json.dumps(info, indent=2))
    log.info("room %s: scale %g, surface %s, %d collider pieces", name, s, surface.describe(), len(colliders))
    return room_dir


def installed_rooms(root: Path | None = None) -> dict[str, dict]:
    out = {}
    root = Path(root or ROOMS_DIR)
    if root.exists():
        for path in sorted(Path(root).glob("*/room.json")):
            try:
                out[path.parent.name] = json.loads(path.read_text())
            except ValueError:
                continue
    return out


def load_room(name: str, root: Path | None = None) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]], tuple]:
    """(room.json, convex pieces [(V, F)], surface slab (V, F)) of an imported room."""
    room_dir = Path(root or ROOMS_DIR) / name
    info = json.loads((room_dir / "room.json").read_text())
    data = np.load(room_dir / "colliders.npz")
    pieces = [(data[f"v{i}"], data[f"f{i}"]) for i in range(info["num_colliders"])]
    return info, pieces, (data["slab_v"], data["slab_f"])


def attribution(info: dict) -> str:
    parts = [f"“{info.get('title') or info.get('name')}”"]
    if info.get("author"):
        parts.append(f"by {info['author']}")
    if info.get("license"):
        parts.append(f"({info['license']})")
    if info.get("source"):
        parts.append(info["source"])
    return " ".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m superdex_quest_teleop.rooms",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOMS_DIR, help="room library (default: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="import a room: a preset, a Sketchfab URL/uid, or a .glb/.gltf/.zip")
    add.add_argument("source", help=f"presets: {', '.join(PRESETS)}")
    add.add_argument("--name", help="room name (default: the preset or file name)")
    add.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN"), help="Sketchfab API token")
    add.add_argument("--surface", type=int, default=0, help="work surface candidate (see `surfaces`)")
    add.add_argument("--scale", type=float, default=None, help="model units -> meters (default: detected)")
    add.add_argument("--no-coacd", action="store_true", help="convex hulls only (faster, coarser)")
    srf = sub.add_parser("surfaces", help="list the work-surface candidates of an imported room")
    srf.add_argument("name")
    sub.add_parser("list", help="list imported rooms")
    sea = sub.add_parser("search", help="search Sketchfab for downloadable rooms")
    sea.add_argument("query", nargs="+")
    sea.add_argument("--count", type=int, default=24)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "search":
        results = sketchfab_search(" ".join(args.query), args.count, os.environ.get("SKETCHFAB_API_TOKEN"))
        for r in results:
            faces = f"{r['faces'] / 1000:.0f}k tris" if r.get("faces") else ""
            print(f"{r['uid']}  {str(r['name'])[:40]:40s} {str(r['author'])[:18]:18s} "
                  f"{str(r['license'] or '')[:22]:22s} {faces:>10s}  {r['url']}")
        print("Import one: python -m superdex_quest_teleop.rooms add <uid> --name my_room"
              "   (under ~500k triangles runs best on the Quest)")
        return
    if args.cmd == "add":
        name = args.name or (args.source if args.source in PRESETS or args.source in installed_rooms(args.root)
                             else re.sub(r"[^A-Za-z0-9_]+", "_", Path(args.source).stem).strip("_").lower())
        import_room(name, args.source, args.token, args.surface, args.scale, not args.no_coacd, args.root)
        args.cmd, args.name = "surfaces", name
    if args.cmd == "surfaces":
        info = installed_rooms(args.root).get(args.name)
        if not info:
            raise SystemExit(f"no room {args.name!r}; import it with `rooms add`")
        print(f"{args.name}: {attribution(info)}")
        print(f"  scale {info['scale']:g} (detected {info['scale_detected']:g}), "
              f"room height {info.get('room_height_m') or float('nan'):.2f} m, {info['num_colliders']} collider pieces")
        for i, text in enumerate(info["surfaces"]):
            mark = "*" if i == info["surface"]["index"] else " "
            print(f"  {mark} [{i}] {text}")
        print(f"Use it: python -m superdex_quest_teleop --environment {args.name}"
              "   (other surface: rooms add NAME --surface N)")
    elif args.cmd == "list":
        for name, info in installed_rooms(args.root).items():
            print(f"{name:20s} {info['surface']['description']}  {attribution(info)}")
        missing = [p for p in PRESETS if p not in installed_rooms(args.root)]
        if missing:
            print("presets not imported yet:", ", ".join(missing))


if __name__ == "__main__":
    main()
