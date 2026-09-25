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

"""Build photoreal teleop hands from SMPL-X + a BEDLAM skin texture.

BEDLAM (and BEDLAM 2.0) render SMPL-X bodies with scanned skin albedo maps in
the SMPL-X UV layout. This tool cuts both hands out of a SMPL-X body (with
optional shape betas), rounds off the wrist, re-skins them to the 25 WebXR
hand joints the teleop client animates, crops the skin texture to the hand
region and writes ``left.glb`` / ``right.glb``:

    python -m superdex_quest_teleop.bedlam_hands \\
        --smplx /path/to/models/smplx/SMPLX_NEUTRAL.npz \\
        --bedlam-root /CT/datasets10/static00/BEDLAM --list
    python -m superdex_quest_teleop.bedlam_hands \\
        --smplx /path/to/SMPLX_NEUTRAL.npz \\
        --bedlam-root /CT/datasets10/static00/BEDLAM --texture-match skin_f_ \\
        --out ~/bedlam_hands
    python -m superdex_quest_teleop --hand-models ~/bedlam_hands

SMPL-X and BEDLAM are licensed by the Max Planck Society for non-commercial
research (registration required); the outputs inherit those terms, so keep
them out of public repositories.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np

from . import hand_skeleton as hs

# SMPL-X joint indices (smplx.joint_names.JOINT_NAMES).
WRIST = {"left": 20, "right": 21}
ELBOW = {"left": 18, "right": 19}
# First joint of each finger chain (3 joints each), SMPL-X order.
FINGER_BASE = {
    "left": {"index": 25, "middle": 28, "pinky": 31, "ring": 34, "thumb": 37},
    "right": {"index": 40, "middle": 43, "pinky": 46, "ring": 49, "thumb": 52},
}
# Fingertip vertices (smplx.vertex_ids['smplx']).
TIP_VERTEX = {
    "left": {"thumb": 5361, "index": 4933, "middle": 5058, "ring": 5169, "pinky": 5286},
    "right": {"thumb": 8079, "index": 7669, "middle": 7794, "ring": 7905, "pinky": 8022},
}
FINGERS = ("index", "middle", "ring", "pinky")
CUT_BEHIND_WRIST = 0.012  # [m] kept behind the wrist joint before the dome
DOME_LENGTH = 0.014       # [m]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def load_smplx(path: Path, betas: np.ndarray | None = None, uv_obj: Path | None = None) -> dict:
    """SMPL-X template (optionally shaped), faces, skinning, joints and UVs."""
    data = dict(np.load(path, allow_pickle=True))
    v = np.asarray(data["v_template"], np.float64)
    if betas is not None and len(betas):
        dirs = np.asarray(data["shapedirs"], np.float64)[:, :, : len(betas)]
        v = v + dirs @ np.asarray(betas, np.float64)
    regressor = data["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    joints = np.asarray(regressor, np.float64) @ v
    model = {
        "v": v,
        "f": np.asarray(data["f"], np.int64),
        "weights": np.asarray(data["weights"], np.float64),
        "joints": joints,
    }
    if uv_obj is not None:
        model["vt"], model["ft"] = read_obj_uv(uv_obj)
    elif "vt" in data and "ft" in data:
        model["vt"] = np.asarray(data["vt"], np.float64)
        model["ft"] = np.asarray(data["ft"], np.int64)
    else:
        raise ValueError(f"{path} has no UVs (vt/ft); pass --uv-obj smplx_uv.obj")
    return model


def read_obj_uv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vt, ft = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            ft.append([int(c.split("/")[1]) - 1 for c in line.split()[1:4]])
    return np.asarray(vt), np.asarray(ft)


def find_textures(root: Path) -> list[Path]:
    """Skin albedo candidates under a BEDLAM checkout."""
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff"):
            continue
        name = path.name.lower()
        if any(k in name for k in ("normal", "_nrm", "rough", "spec", "disp", "mask", "_ao")):
            continue
        if any(k in name for k in ("skin", "_alb", "albedo", "diffuse", "basecolor", "base_color")):
            found.append(path)
    return found


# ----------------------------------------------------------------------------
# Hand extraction
# ----------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def hand_joints(model: dict, side: str) -> np.ndarray:
    """The 25 WebXR joint positions of one SMPL-X hand."""
    j = model["joints"]
    out = np.zeros((hs.NUM_JOINTS, 3))
    wrist = j[WRIST[side]]
    out[hs.WRIST] = wrist
    t = FINGER_BASE[side]["thumb"]
    out[1:4] = j[t:t + 3]
    out[4] = model["v"][TIP_VERTEX[side]["thumb"]]
    for f, finger in enumerate(FINGERS):
        base = hs.INDEX_METACARPAL + 5 * f
        k = FINGER_BASE[side][finger]
        proximal = j[k]
        # Metacarpal joints sit in the palm, about a third of the way out.
        out[base] = wrist + 0.3 * (proximal - wrist)
        out[base + 1:base + 4] = j[k:k + 3]
        out[base + 4] = model["v"][TIP_VERTEX[side][finger]]
    return out


def bind_frames(joints: np.ndarray, side: str) -> np.ndarray:
    """(25, 4, 4) world-from-bone bind matrices in WebXR convention
    (-Z along the bone toward the fingertip, +Y dorsal)."""
    palm = np.cross(joints[21] - joints[0], joints[6] - joints[0])
    dorsal = _normalize(palm if side == "right" else -palm)
    frames = np.zeros((hs.NUM_JOINTS, 4, 4))
    for i in range(hs.NUM_JOINTS):
        if i == hs.WRIST:
            nxt = joints[hs.MIDDLE_PROXIMAL]
            forward = nxt - joints[i]
        elif i in hs.FINGER_TIPS:
            forward = joints[i] - joints[i - 1]
        else:
            forward = joints[i + 1] - joints[i]
        z = -_normalize(forward)
        y = _normalize(dorsal - z * np.dot(dorsal, z))
        x = np.cross(y, z)
        frames[i] = np.eye(4)
        frames[i][:3, :3] = np.stack([x, y, z], axis=1)
        frames[i][:3, 3] = joints[i]
    return frames


def _smplx_to_webxr_bone(side: str) -> dict[int, int]:
    """SMPL-X joint -> WebXR bone index for skin weights."""
    mapping = {WRIST[side]: hs.WRIST, ELBOW[side]: hs.WRIST}
    t = FINGER_BASE[side]["thumb"]
    for k in range(3):
        mapping[t + k] = 1 + k
    for f, finger in enumerate(FINGERS):
        base = hs.INDEX_METACARPAL + 5 * f
        for k in range(3):
            mapping[FINGER_BASE[side][finger] + k] = base + 1 + k
    return mapping


def extract_hand(model: dict, side: str) -> dict:
    """Cut one hand at the wrist, cap it with a dome, split vertices by UV."""
    v, f, weights = model["v"], model["f"], model["weights"]
    joints = hand_joints(model, side)
    wrist = joints[hs.WRIST]
    elbow = model["joints"][ELBOW[side]]
    axis = _normalize(wrist - elbow)  # forearm direction, toward the hand
    mapping = _smplx_to_webxr_bone(side)
    hand_weight = weights[:, list(mapping)].sum(axis=1)
    along = (v - wrist) @ axis
    keep_vertex = (along > -CUT_BEHIND_WRIST) & (hand_weight > 0.2)
    keep_face = keep_vertex[f].all(axis=1)
    faces = f[keep_face]
    faces_uv = model["ft"][keep_face]

    # glTF vertices are (position, uv) pairs.
    pairs = {}
    positions, uvs, sources = [], [], []
    indices = np.zeros(faces.shape, np.int64)
    for fi in range(len(faces)):
        for c in range(3):
            key = (int(faces[fi, c]), int(faces_uv[fi, c]))
            idx = pairs.get(key)
            if idx is None:
                idx = len(positions)
                pairs[key] = idx
                positions.append(v[key[0]])
                uvs.append(model["vt"][key[1]])
                sources.append(key[0])
            indices[fi, c] = idx
    positions = np.asarray(positions)
    uvs = np.asarray(uvs)
    sources = np.asarray(sources)

    # Skin weights on the 25 WebXR bones (top 4, normalized).
    bone_weights = np.zeros((len(positions), hs.NUM_JOINTS))
    for smplx_joint, bone in mapping.items():
        bone_weights[:, bone] += weights[sources, smplx_joint]
    bone_weights[:, hs.WRIST] += np.clip(1.0 - bone_weights.sum(axis=1), 0.0, None)

    # Cap the wrist: boundary edges of the cut, fanned to a dome.
    positions, uvs, bone_weights, indices = _cap_wrist(
        positions, uvs, bone_weights, indices, sources, wrist, axis
    )
    order = np.argsort(-bone_weights, axis=1)[:, :4]
    top = np.take_along_axis(bone_weights, order, axis=1)
    top /= np.maximum(top.sum(axis=1, keepdims=True), 1e-9)
    return {
        "positions": positions,
        "uvs": uvs,
        "indices": indices,
        "joints": order.astype(np.uint16),
        "weights": top,
        "bind": bind_frames(joints, side),
    }


def _cap_wrist(positions, uvs, bone_weights, indices, sources, wrist, axis):
    from collections import Counter

    # Boundary edges between welded positions, so UV seams don't count.
    _, src_of = np.unique(np.round(positions, 6), axis=0, return_inverse=True)
    src_of = src_of.reshape(-1)
    edges = Counter()
    directed = {}
    for tri in indices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((src_of[a], src_of[b])))
            edges[key] += 1
            directed[(src_of[a], src_of[b])] = (a, b)
    boundary = [(a, b) for (a, b), (ia, ib) in directed.items()
                if edges[tuple(sorted((a, b)))] == 1]
    if not boundary:
        return positions, uvs, bone_weights, indices
    # Only the loop at the cut (the rest of the hand is closed).
    first = {}
    for a, b in boundary:
        first[a] = b
    start = max(first, key=lambda s: -((positions[np.where(src_of == s)[0][0]] - wrist) @ axis))
    loop, cursor = [start], first[start]
    while cursor != start and cursor in first and len(loop) < 10000:
        loop.append(cursor)
        cursor = first[cursor]
    ring_idx = [directed[(loop[k], loop[(k + 1) % len(loop)])][0] for k in range(len(loop))]
    ring = positions[ring_idx]
    center = ring.mean(axis=0)
    n_rings = 4
    new_pos, new_uv, new_w = [], [], []
    tri = []
    prev = ring_idx
    base_uv = uvs[ring_idx].mean(axis=0)
    for r in range(1, n_rings + 1):
        t = r / (n_rings + 1)
        shrink = np.sqrt(max(0.0, 1.0 - t * t))
        layer = []
        for k, idx in enumerate(ring_idx):
            p = center + (ring[k] - center) * shrink - axis * DOME_LENGTH * t
            layer.append(len(positions) + len(new_pos))
            new_pos.append(p)
            new_uv.append(uvs[idx])
            new_w.append(bone_weights[idx])
        for k in range(len(ring_idx)):
            a, b = prev[k], prev[(k + 1) % len(prev)]
            c, d = layer[k], layer[(k + 1) % len(layer)]
            tri += [[b, a, c], [b, c, d]]
        prev = layer
    tip = len(positions) + len(new_pos)
    new_pos.append(center - axis * DOME_LENGTH)
    new_uv.append(base_uv)
    new_w.append(bone_weights[ring_idx].mean(axis=0))
    for k in range(len(prev)):
        tri.append([prev[(k + 1) % len(prev)], prev[k], tip])
    tri = np.asarray(tri)
    # Orient the cap outward (away from the palm).
    all_pos = np.concatenate([positions, np.asarray(new_pos)])
    n = np.cross(all_pos[tri[0, 1]] - all_pos[tri[0, 0]], all_pos[tri[0, 2]] - all_pos[tri[0, 0]])
    if n @ -axis < 0:
        tri = tri[:, ::-1]
    return (
        all_pos,
        np.concatenate([uvs, np.asarray(new_uv)]),
        np.concatenate([bone_weights, np.asarray(new_w)]),
        np.concatenate([indices, tri]),
    )


def vertex_normals(positions: np.ndarray, indices: np.ndarray, sources_key: np.ndarray | None = None) -> np.ndarray:
    normals = np.zeros_like(positions)
    tri = positions[indices]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for c in range(3):
        np.add.at(normals, indices[:, c], face_n)
    # Share normals across UV seams (same position).
    keys = np.round(positions, 7)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    summed = np.zeros((inverse.max() + 1, 3))
    np.add.at(summed, inverse.reshape(-1), normals)
    normals = summed[inverse.reshape(-1)]
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------
# Texture and GLB
# ----------------------------------------------------------------------------


def crop_texture(image_path: Path, uvs: np.ndarray, max_size: int = 2048) -> tuple[bytes, np.ndarray]:
    """Crop the atlas to the hand's UV region; returns (JPEG bytes, new UVs)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    lo = np.clip(uvs.min(axis=0) - 0.01, 0.0, 1.0)
    hi = np.clip(uvs.max(axis=0) + 0.01, 0.0, 1.0)
    # UV v runs bottom-up; image rows top-down.
    box = (int(lo[0] * w), int((1 - hi[1]) * h), int(np.ceil(hi[0] * w)), int(np.ceil((1 - lo[1]) * h)))
    crop = image.crop(box)
    scale = min(1.0, max_size / max(crop.size))
    if scale < 1.0:
        crop = crop.resize((max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale))), Image.LANCZOS)
    new_uv = np.stack([
        (uvs[:, 0] * w - box[0]) / (box[2] - box[0]),
        ((1 - uvs[:, 1]) * h - box[1]) / (box[3] - box[1]),
    ], axis=1)  # glTF UVs: origin top-left
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), new_uv


def write_glb(path: Path, hand: dict, texture_jpeg: bytes | None, uvs_gltf: np.ndarray) -> None:
    """A skinned glTF binary: flat bones named after the WebXR joints."""
    positions = hand["positions"].astype(np.float32)
    normals = vertex_normals(hand["positions"], hand["indices"]).astype(np.float32)
    blobs: list[bytes] = []
    views, accessors = [], []

    def add(array: np.ndarray, gltf_type: str, component: int, target: int | None = None, minmax=False):
        data = np.ascontiguousarray(array).tobytes()
        offset = sum(len(b) for b in blobs)
        pad = (-len(data)) % 4
        blobs.append(data + b"\0" * pad)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": int(array.shape[0]),
               "type": gltf_type}
        if minmax:
            acc["min"] = array.min(axis=0).tolist()
            acc["max"] = array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(positions, "VEC3", 5126, 34962, minmax=True)
    a_nrm = add(normals, "VEC3", 5126, 34962)
    a_uv = add(uvs_gltf.astype(np.float32), "VEC2", 5126, 34962)
    a_jnt = add(hand["joints"].astype(np.uint16), "VEC4", 5123, 34962)
    a_wgt = add(hand["weights"].astype(np.float32), "VEC4", 5126, 34962)
    a_idx = add(hand["indices"].astype(np.uint32).reshape(-1), "SCALAR", 5125, 34963)
    inverse = np.linalg.inv(hand["bind"]).transpose(0, 2, 1).astype(np.float32)  # column-major
    a_ibm = add(inverse.reshape(-1, 16), "MAT4", 5126)

    nodes = []
    for i, name in enumerate(hs.JOINT_NAMES):
        m = hand["bind"][i]
        from .hand_model import matrix_to_quat

        nodes.append({"name": name, "translation": m[:3, 3].tolist(),
                      "rotation": matrix_to_quat(m[:3, :3]).tolist()})
    mesh_node = len(nodes)
    nodes.append({"name": "hand_mesh", "mesh": 0, "skin": 0})
    nodes.append({"name": "Armature", "children": list(range(hs.NUM_JOINTS)) + [mesh_node]})
    material = {"name": "skin", "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.55,
                                                          "baseColorFactor": [1, 1, 1, 1]}}
    doc = {
        "asset": {"version": "2.0", "generator": "superdex_quest_teleop.bedlam_hands"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                                                   "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt},
                                    "indices": a_idx, "material": 0}]}],
        "skins": [{"joints": list(range(hs.NUM_JOINTS)), "inverseBindMatrices": a_ibm}],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": views,
    }
    if texture_jpeg is not None:
        offset = sum(len(b) for b in blobs)
        pad = (-len(texture_jpeg)) % 4
        blobs.append(texture_jpeg + b"\0" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(texture_jpeg)})
        doc["images"] = [{"bufferView": len(views) - 1, "mimeType": "image/jpeg"}]
        doc["textures"] = [{"source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    binary = b"".join(blobs)
    doc["buffers"] = [{"byteLength": len(binary)}]
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * ((-len(text)) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, total))
        out.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        out.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def build(model: dict, texture: Path | None, out_dir: Path, max_texture: int = 2048) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for side in hs.SIDES:
        hand = extract_hand(model, side)
        if texture is not None:
            jpeg, uvs = crop_texture(texture, hand["uvs"], max_texture)
        else:
            jpeg, uvs = None, np.stack([hand["uvs"][:, 0], 1 - hand["uvs"][:, 1]], axis=1)
        path = out_dir / f"{side}.glb"
        write_glb(path, hand, jpeg, uvs)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smplx", type=Path, required=True, help="SMPL-X model (.npz, e.g. SMPLX_NEUTRAL.npz)")
    parser.add_argument("--uv-obj", type=Path, help="SMPL-X UV template .obj if the model has no vt/ft")
    parser.add_argument("--bedlam-root", type=Path, help="BEDLAM / BEDLAM 2.0 directory to search for skin textures")
    parser.add_argument("--texture", type=Path, help="skin albedo in SMPL-X UV layout")
    parser.add_argument("--texture-match", help="pick the first found texture whose path contains this")
    parser.add_argument("--list", action="store_true", help="list skin textures under --bedlam-root and exit")
    parser.add_argument("--betas", type=float, nargs="*", default=None, help="SMPL-X shape coefficients")
    parser.add_argument("--max-texture", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("bedlam_hands"))
    args = parser.parse_args()

    texture = args.texture
    if args.bedlam_root is not None and texture is None:
        found = find_textures(args.bedlam_root)
        if args.list or not found:
            for i, path in enumerate(found):
                print(f"{i:4d}  {path}")
            if not found:
                print(f"no skin albedo textures found under {args.bedlam_root}")
            return
        if args.texture_match:
            found = [p for p in found if args.texture_match in str(p)] or found
        texture = found[0]
    model = load_smplx(args.smplx, np.asarray(args.betas) if args.betas else None, args.uv_obj)
    for path in build(model, texture, args.out, args.max_texture):
        print(f"wrote {path}")
    print(f"texture: {texture}")
    print(f"use with: python -m superdex_quest_teleop --hand-models {args.out}")


if __name__ == "__main__":
    main()
