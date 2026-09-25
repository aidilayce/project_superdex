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

"""A synthetic stand-in for a SMPL-X model file (the real one is licensed).

Both hands come from the WebXR generic-hand mesh (placed at the SMPL-X hand
joints, with its skin weights mapped back to SMPL-X joints); the file has
the same keys, joint indices and fingertip vertex slots as SMPLX_NEUTRAL.npz.
"""

import json
import struct

import numpy as np

from superdex_quest_teleop import bedlam_hands as bh
from superdex_quest_teleop import hand_skeleton as hs
from superdex_quest_teleop.hand_display import GENERIC_HAND_DIR, bind_pose

NUM_VERTS = 10475
NUM_JOINTS = 55


def _mesh(path):
    data = path.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    off = 20 + length
    blob = data[off + 8:]
    prim = doc["meshes"][0]["primitives"][0]
    skin_joints = doc["skins"][0]["joints"]

    def acc(i):
        a = doc["accessors"][i]
        view = doc["bufferViews"][a["bufferView"]]
        dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
        k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(blob, dtype, a["count"] * k, view.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], k)

    names = [doc["nodes"][n]["name"] for n in skin_joints]
    joints = acc(prim["attributes"]["JOINTS_0"]).astype(int)
    return (acc(prim["attributes"]["POSITION"]).astype(np.float64), acc(prim["indices"]).reshape(-1, 3).astype(int),
            acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64), joints, acc(prim["attributes"]["WEIGHTS_0"]),
            names)


def make(path, offset=0.35):
    v = np.zeros((NUM_VERTS, 3))
    weights = np.zeros((NUM_VERTS, NUM_JOINTS))
    weights[:, 0] = 1.0
    faces, vt, ft = [], [], []
    used = 0
    joint_pos = np.zeros((NUM_JOINTS, 3))
    for side in hs.SIDES:
        glb = GENERIC_HAND_DIR / f"{side}.glb"
        pos, tri, uv, jidx, jw, names = _mesh(glb)
        bind = bind_pose(glb)
        shift = np.array([offset if side == "right" else -offset, 1.2, 0.0])
        pos = pos + shift
        # SMPL-X joints from the model's bind pose.
        smplx_of = {}
        joint_pos[bh.WRIST[side]] = bind["wrist"][:3, 3] + shift
        joint_pos[bh.ELBOW[side]] = bind["wrist"][:3, 3] + shift + bind["wrist"][:3, 2] * 0.25
        base = bh.FINGER_BASE[side]
        for k, part in enumerate(("metacarpal", "phalanx-proximal", "phalanx-distal")):
            joint_pos[base["thumb"] + k] = bind[f"thumb-{part}"][:3, 3] + shift
            smplx_of[f"thumb-{part}"] = base["thumb"] + k
        smplx_of["thumb-tip"] = base["thumb"] + 2
        for finger in bh.FINGERS:
            for k, part in enumerate(("phalanx-proximal", "phalanx-intermediate", "phalanx-distal")):
                joint_pos[base[finger] + k] = bind[f"{finger}-finger-{part}"][:3, 3] + shift
                smplx_of[f"{finger}-finger-{part}"] = base[finger] + k
            smplx_of[f"{finger}-finger-tip"] = base[finger] + 2
            smplx_of[f"{finger}-finger-metacarpal"] = bh.WRIST[side]
        smplx_of["wrist"] = bh.WRIST[side]
        n = len(pos)
        ids = np.arange(used, used + n)
        v[ids] = pos
        # Fingertip positions in the fingertip vertex slots of the real model.
        for finger in ("thumb",) + bh.FINGERS:
            name = "thumb-tip" if finger == "thumb" else f"{finger}-finger-tip"
            v[bh.TIP_VERTEX[side][finger]] = bind[name][:3, 3] + shift
        w = np.zeros((n, NUM_JOINTS))
        for k in range(4):
            for i in range(n):
                w[i, smplx_of[names[jidx[i, k]]]] += jw[i, k]
        weights[ids] = w
        faces.append(ids[tri])
        vt.append(uv * [1, -1] + [0, 1])  # glTF UV -> OBJ-style (bottom-up)
        ft.append(tri + sum(len(x) for x in vt[:-1]))
        used += n
    # One-hot regressor on vertices placed at the joints.
    regressor = np.zeros((NUM_JOINTS, NUM_VERTS))
    for j in range(NUM_JOINTS):
        slot = NUM_VERTS - 1 - j
        v[slot] = joint_pos[j]
        regressor[j, slot] = 1.0
    np.savez(path, v_template=v, f=np.concatenate(faces), weights=weights, J_regressor=regressor,
             vt=np.concatenate(vt), ft=np.concatenate(ft), shapedirs=np.zeros((NUM_VERTS, 3, 10)),
             kintree_table=np.zeros((2, NUM_JOINTS), int))
    return path
