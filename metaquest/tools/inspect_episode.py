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

"""Summarize a recorded teleoperation episode and show how to use its
per-point contacts: per-step contact points, and the force each hand link
applies to each object.

    python metaquest/tools/inspect_episode.py recordings/cube_20260101_120000.h5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import h5py
import numpy as np


def contacts_at(f: h5py.File, step: int) -> dict[str, np.ndarray]:
    """All contact point fields of one step (CSR slice)."""
    offsets = f["contacts/step_offset"]
    s = slice(int(offsets[step]), int(offsets[step + 1]))
    return {name: f[f"contacts/{name}"][s] for name in f["contacts"] if name != "step_offset"}


def pair_forces(f: h5py.File, step: int) -> dict[tuple[int, int], np.ndarray]:
    """Net force [N] on actor A from actor B at one step, for every touching
    pair. A point owned by B pushes on A with the opposite force."""
    c = contacts_at(f, step)
    out: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(3))
    for owner, other, force in zip(c["owner"], c["other"], c["force"]):
        out[(int(owner), int(other))] += force
        out[(int(other), int(owner))] -= force
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode")
    args = parser.parse_args()
    with h5py.File(args.episode) as f:
        names = f["actors/name"][:].astype(str)
        sides = f["actors/hand_side"][:].astype(str)
        steps = int(f.attrs["num_steps"])
        dt = float(f.attrs["time_step"])
        meta = json.loads(f.attrs["metadata"])
        print(f"{args.episode}")
        print(f"  scene      {f.attrs['scene_name']} ({f.attrs['scene_id']})")
        print(f"  duration   {steps} steps x {dt * 1000:.1f} ms = {steps * dt:.2f} s")
        print(f"  contacts   {int(f.attrs['num_contacts'])} points ({meta.get('contact_mode')} mode)")
        for side in f["hands"]:
            tracked = f[f"hands/{side}/tracked"][:]
            print(f"  {side:5s} hand tracked {tracked.mean() * 100:.0f}% of steps")
        objects = [i for i, s in enumerate(sides) if not s and not f["actors/is_static"][i]]
        print(f"  objects    {', '.join(names[i] for i in objects)}")

        # Peak hand -> object forces, per hand link.
        peak: dict[tuple[str, str], float] = defaultdict(float)
        for t in range(steps):
            for (a, b), force in pair_forces(f, t).items():
                if a in objects and b >= 0 and sides[b]:
                    key = (names[b], names[a])
                    peak[key] = max(peak[key], float(np.linalg.norm(force)))
        if peak:
            print("  peak force applied by hand links to objects:")
            for (link, obj), value in sorted(peak.items(), key=lambda kv: -kv[1])[:12]:
                print(f"    {link:44s} -> {obj:24s} {value:7.3f} N")
        for key in f.get("deformables", {}):
            g = f[f"deformables/{key}"]
            n = len(g["node_index"])
            print(f"  deformable {g.attrs['name']}: {g['surface_positions'].shape[1]} surface nodes, "
                  f"{n} node-force samples")


if __name__ == "__main__":
    main()
