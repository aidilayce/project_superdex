# Meta Quest hand-tracking teleoperation and contact data collection

This example is the Meta Quest counterpart of the Apple Vision Pro example
(`visionos/` on the `benbiggs/visionpro_anydex_handtracking` branch). The
operator wears a Quest headset and manipulates SuperDex objects with two
physical **Meta XR hands**. Hand tracking comes from the Quest. The
simulation runs on a PC. Every step is recorded to disk, including every
**contact point with its force**, so a teleoperation session produces
synthetic manipulation data. It follows the same idea as Isaac Sim's
[teleop synthetic data generation tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/synthetic_data_generation/tutorial_replicator_teleop_sdg.html).

The Vision Pro example has two scenes (Box and Blocks, Free Rope). This one
builds its scenes from the whole SuperDex asset library: every prefab in
`assets/prefabs/`, plus rigid and deformable objects from
`superdex_physics/assets` (soft duck, soft cube, rope, t-shirt cloth). A new
prefab added to `assets/prefabs/` shows up as a scene automatically.

```text
 Meta Quest (browser, WebXR)                     PC running SuperDex (Python)
 ───────────────────────────                     ─────────────────────────────────────────
 XRHand: 25 joints / hand  ──── WebSocket ────▶  retarget → Meta XR hand bot (27 DoF / hand)
 (joint poses + radii,          (JSON, 72 Hz)    pose controller: wrist spring + joint PD
  head pose, in the                              SuperDex contact solver (fixed step)
  physics frame)                                 contact/force queries ──▶ HDF5 episode
 three.js renderer  ◀────────── WebSocket ─────  actor poses, deformable vertices,
 in-VR button panel             (binary, 60 Hz)  strongest contacts
```

Unlike the visionOS example, nothing is installed on the headset. The Quest
browser opens a page served by the PC. That page streams the hand skeletons
and renders the scene the PC sends back. The same page in a desktop browser
is a spectator view with the same controls.

## Requirements

* A PC with the SuperDex requirements (Linux, Windows or macOS, Python 3.12).
  More CPU cores mean more scenes run in real time.
* Meta Quest 2, 3, 3S or Pro. Hand tracking must be on
  (*Settings → Movement tracking → Hand and body tracking*).
* To connect over USB (recommended): a USB-C cable and `adb` (Android
  platform tools), with the headset in developer mode.

## Install

From the repository root:

```bash
uv venv
uv pip install superdex aiohttp h5py cryptography   # or: uv pip install -e metaquest
cd metaquest
uv run --no-project python -m superdex_quest_teleop --list-scenes
```

The PC serves everything the headset needs, including a bundled copy of
three.js, so the headset doesn't need internet access.

## Run

```bash
cd metaquest
uv run --no-project python -m superdex_quest_teleop --scene duck_lamp --out ../recordings
```

The server listens on two ports and prints the URLs to open:

* `http://…:8080`: the desktop viewer, and the Quest over USB.
* `https://…:8443`: the Quest over Wi-Fi. The server creates a self-signed
  certificate for this PC's IP addresses on first use.

WebXR, and therefore hand tracking and the **Enter VR** button, only works on
`https://` pages or on `http://localhost`. If the Quest opens
`http://<pc-ip>:8080`, it is redirected to the HTTPS address.

**Over Wi-Fi.** On the Quest, open `https://<pc-ip>:8443/`. The first time,
the browser warns about the certificate: tap *Advanced → Proceed*. Then tap
**Enter VR**, or **Enter passthrough** on Quest 3 to overlay the table on your
room. Allow hand tracking when the browser asks, and put the controllers
down. Use 5 GHz or 6 GHz Wi-Fi.

**Over USB (lowest latency, no certificate).**

```bash
adb reverse tcp:8080 tcp:8080
```

Then open `http://localhost:8080/` in the Quest browser.

**Without a headset.** Open `http://localhost:8080/` on the PC to watch. Add
`--synthetic` and a scripted Quest-format hand reaches, grasps, lifts and
releases the object, which exercises the whole pipeline.

### In the headset

When the session starts, the virtual table (the physics origin) is placed
0.42 m in front of your eyes and 0.5 m below them, facing where you look. It
is drawn as a butcher-block kitchen island. Your real hands appear as small
blue joint spheres ("ghost"). The simulated Meta XR hands follow them
physically: they stop at objects instead of passing through, so what you see
matches the forces being recorded.

A button panel floats to the left of the table. Poke a button with an index
fingertip:

| Button | Action |
| :-- | :-- |
| ● Rec / ■ Stop | Start or stop recording an episode |
| Reset | Rebuild the current scene |
| ◀ Scene / Scene ▶ | Previous or next scene |
| Cam view | Put the table back in front of you (re-center on your current view) |
| Table ▲ / ▼ | Raise or lower the table by 3 cm |
| Contacts | Show or hide contact points and force vectors |
| Ghost | Show or hide your tracked joints |
| Hands | Switch between skinned hands and the Meta XR robot-hand meshes |
| Exit VR | Leave the immersive session |

The panel also shows the scene, the simulation's real-time factor, the
headset frame rate and the recording state. It warns when no hands are
tracked, for example while you still hold the controllers.

**Desktop view:** a scene menu, **Reset**, **Record** and **Cam view**. Cam
view looks through the operator's eyes: it follows the headset live while one
is connected; otherwise it shows the default operator viewpoint. Dragging the
view returns to orbiting.

Keys: `Space` records, `R` resets, `V` toggles Cam view, `H` switches the
hand view, `N`/`P` changes scene, `C` toggles contacts.

### Look: scenes, lighting and hands

**Where realism comes from.** The realistic videos in the SuperDex README were
rendered in Unreal Engine 5 with *SuperDex Teleop*, which is not released yet
(announced for Q4 2026); the public repository ships no Unreal scenes. This
client uses everything the repository does ship, plus CC0/MIT assets:

* **Objects:** every prefab is drawn with its upstream textured PBR render
  model (`assets/prefabs/*/render/*.glb`: the wooden box and blocks,
  pegboards, printed paper cups, shape sorter, chain, sphere). The physics
  still uses the collision meshes.
* **Lighting:** image-based lighting from SuperDex Studio's HDR
  (`superdex_studio/assets/ibl/studio_small_08_1k.hdr`, Poly Haven, CC0).
* **Kitchen:** with the default `--environment auto`, the PC downloads a CC0
  kitchen HDRI from [Poly Haven](https://polyhaven.com/hdris/indoor) once
  (cached in `~/.superdex_quest_teleop/environments`). It becomes both the
  photographic 360° backdrop and the light that falls on the objects and
  hands. Until it's available (or offline), a procedural kitchen is shown.
  Pick a different HDRI with
  `python -m superdex_quest_teleop.environment --list` and `--id <asset>`
  (add `--resolution 4k` for a sharper backdrop).
* **Your own kitchen:** pass `--environment` a 360° photo of your kitchen
  (`.jpg`/`.png`, taken from where you stand), an `.hdr`, or a `.glb`/`.gltf`
  model (origin on the floor under the table center, -Z facing away from you).
  On Quest 3, **Enter passthrough** shows your real room.
* **Hands:** each hand is one continuous skinned mesh (the WebXR
  `generic-hand` model, MIT). Every frame the PC turns the *simulated* Meta
  XR hand's link poses into the 25 WebXR joint poses the mesh is rigged to,
  so the skin shows exactly where physics put the fingers. The skin has a
  UV-mapped albedo, a normal map with pores and creases, a subtle sheen and
  fingernails. The forearm fades out toward the elbow. Press **H** (or poke
  *Hands*) to see the Meta XR robot-hand meshes that actually collide.

**BEDLAM-style hands.** BEDLAM renders SMPL-X bodies with scanned skin
textures in Unreal. Those textures are licensed for registered,
non-commercial research use, so they are not bundled here. You can use them
(or any textured hand) with `--hand-models DIR`, where `DIR` holds `left.glb`
and `right.glb`:

1. Rig the hand mesh with 25 bones named like the WebXR joints (`wrist`,
   `thumb-metacarpal`, …, `pinky-finger-tip`). Orient each bone -Z along the
   bone toward the fingertip, +Y on the back of the hand. In Blender you can
   start from `web/vendor/webxr-input-profiles/generic-hand/*.glb` and
   transfer weights to your mesh.
2. Bake or assign the albedo, normal and roughness textures (for example a
   BEDLAM/SMPL-X skin texture) to its UVs, and export as glTF binary.

Models that bring their own textures are shown with their own materials.

### Troubleshooting

| Symptom | Fix |
| :-- | :-- |
| **Enter VR** shows "WebXR only works on secure pages" | Open the `https://<pc-ip>:8443/` address (accept the certificate), or use USB with `adb reverse` and `http://localhost:8080/` |
| The certificate page keeps coming back | The PC's IP changed. Restart the server: it re-issues the certificate for the new address |
| In VR, but the simulated hands don't move | Put the controllers down. Turn on *Settings → Movement tracking → Hand and body tracking*. Allow hand tracking when the browser asks. The panel says "No hands" or "Put the controllers down" when this is the problem |
| Hands move, but jerkily or late | Use USB or 5/6 GHz Wi-Fi. Check the RTF in the HUD: below 1 means the PC can't simulate the scene in real time |
| "Not connected to the PC" | The PC firewall must allow TCP 8080 and 8443 |

### Options

| Flag | Default | Meaning |
| :-- | :-- | :-- |
| `--scene ID` | `box_and_blocks` | Initial scene (`--list-scenes`) |
| `--out DIR` | `recordings` | Where episodes are written |
| `--contacts hand\|all` | `hand` | `hand`: every contact involving a hand link. `all`: also object/object and object/table contacts |
| `--min-contact-force N` | `0` | Drop contact points with smaller force. `0` also keeps zero-force near-contact samples |
| `--keep-self-contacts` | off | Also record contacts between links of the same hand |
| `--hand-mesh lowpoly\|highpoly` | `lowpoly` | Meta XR hand collision mesh |
| `--record` | off | Start recording immediately |
| `--synthetic` | off | Scripted right hand instead of the headset |
| `--http-port N` / `--https-port N` | `8080` / `8443` | Listening ports |
| `--no-https` | off | Serve plain HTTP only |
| `--environment auto\|procedural\|FILE` | `auto` | `auto`: CC0 kitchen HDRI from Poly Haven. `procedural`: the built-in kitchen. `FILE`: your `.hdr`, 360° photo or `.glb` |
| `--hand-models DIR` | WebXR generic hand | Rigged, textured `left.glb`/`right.glb` (e.g. BEDLAM-style hands) |
| `--threads N` | `-1` | SuperDex worker threads |

## Scenes

The table is a static plane at y = 0. Z-up prefabs are rotated into the
Y-up physics frame (as in the visionOS prefab gallery) and centered on the
table using their measured bounds.

| id | Kind | Content |
| :-- | :-- | :-- |
| `box_and_blocks` | rigid | Box and Blocks test (32 blocks) |
| `duck_lamp` | soft | Neo-Hookean duck lamp |
| `sphere` | rigid | 3 cm sphere |
| `shape_box` | rigid | Shape sorter with lid (11 shapes). Heavy |
| `nine_hole_peg_test` | rigid | Nine hole peg test |
| `functional_dexterity_test` | rigid | Functional dexterity test (16 pegs) |
| `fdt_peg`, `paper_cup`, `block_red` | rigid | Single objects |
| `paper_cup_pyramid` | rigid | 10-cup pyramid |
| `chain` | rigid | 10-link chain hanging from a fixed link (1/120 s step). Heavy |
| `cube` | rigid | 5.7 cm puzzle-cube-sized block |
| `soft_cube`, `soft_duck` | soft | Foam cube, soft duck |
| `free_rope` | rod | The visionOS free rope |
| `cloth` | shell | T-shirt with self contact (1/30 s step). Heavy |
| `medley` | mixed | Sphere, cup, block, peg, cube and a soft duck |

To add a scene, drop a prefab into `assets/prefabs/<name>/`, or add a
`SceneSpec` to `superdex_quest_teleop/scenes.py`.

The loop runs at the scene's fixed step and paces itself to wall-clock time.
If a step takes longer than its duration, the simulation runs in slow motion
rather than skipping steps. The HUD shows the real-time factor (RTF).

These are real-time factors measured on a 4-core cloud VM with one hand
grasping and recording on:

* **Real time:** `sphere`, `cube`, `paper_cup`, `nine_hole_peg_test`,
  `functional_dexterity_test`.
* **0.8–0.9×:** `soft_cube`, `free_rope`, `soft_duck`.
* **0.6–0.8×:** `box_and_blocks`, `duck_lamp`, `medley`.
* **Much slower than real time:** `paper_cup_pyramid`, `shape_box`, `chain`,
  `cloth`.

More cores help. Recording costs a few microseconds per contact point, so
contact-dense moments slow the loop down while recording.

## Recorded data

Each episode is one HDF5 file, `<out>/<scene>_<timestamp>.h5`. With T steps,
N actors (objects, the table and 19 links per hand) and K contact points in
total:

| Dataset | Shape | Content |
| :-- | :-- | :-- |
| `actors/name`, `kind`, `hand_side`, `is_static`, `mass` | N | Actor table. `hand_side` is `left`/`right` for hand links |
| `steps/sim_time`, `wall_time`, `step` | T | Timing |
| `actors/pose` | T×N×7 | World pose `[px py pz qx qy qz qw]` |
| `actors/lin_vel`, `ang_vel` | T×N×3 | Rigid body velocities |
| `actors/contact_force` | T×N×3 | Total contact force on each actor [N] |
| `hands/<side>/joints` | T×25×3 | Quest hand joints (WebXR order), physics frame |
| `hands/<side>/joint_rotations`, `joint_radii` | T×25×4, T×25 | Joint orientations and radii from the runtime |
| `hands/<side>/tracked`, `retarget_residual` | T | Tracking state; retargeting RMS error [m] |
| `hands/<side>/target_qpos` | T×27 | Retargeted Meta XR hand DoFs (names in the metadata) |
| `hands/<side>/link_pose` | T×19×7 | Simulated hand link poses |
| `head/pose` | T×7 | Headset pose, physics frame |
| `contacts/step_offset` | T+1 | Contacts of step t are rows `step_offset[t]:step_offset[t+1]` |
| `contacts/owner`, `other` | K | Actor indices of the two touching actors (`-1` if unknown) |
| `contacts/pos_owner`, `pos_other` | K×3 | Contact point on the owner, and the closest point on the other actor |
| `contacts/normal` | K×3 | Contact normal, pointing away from `other` |
| `contacts/force` | K×3 | Force on `owner` from `other` [N], already weighted by `int_weight` |
| `contacts/vel_owner`, `vel_other` | K×3 | Point velocities |
| `contacts/distance` | K | Signed separation (negative when penetrating) |
| `contacts/int_weight` | K | Surface area the sample represents [m²] |
| `contacts/element`, `sample`, `barycentric` | K, K, K×3 | Surface triangle, sample index and barycentric coordinates on the owner's mesh |
| `deformables/<i>/surface_positions` | T×V×3 | Deformed surface mesh, world frame (`surface_triangles` gives the faces) |
| `deformables/<i>/node_index`, `node_force`, `node_force_offset` | M, M×3, T+1 | Per-node contact forces on soft bodies |

The root attributes hold the scene id and name, the time step and a JSON
`metadata` string. The metadata includes the hand link and DoF names, the
Quest joint names and the recording options.

Contact samples live on one actor's surface. Which actor owns a sample
depends on the collider pair: for example, a rigid link against a soft body is
sampled on the soft body. The recorder therefore queries hands and objects
alike. To get the net force on actor A from actor B, add the forces of the
points owned by A with `other == B`, and subtract the forces of the points
owned by B with `other == A`. The test suite checks that these sums reproduce
the engine's total contact force on each object. `tools/inspect_episode.py`
shows how to read an episode and reports the peak force each hand link
applies to each object:

```bash
python metaquest/tools/inspect_episode.py recordings/cube_20260101_120000.h5
```

## Retargeting

The Meta XR hand bot (`assets/bots/hands/oculus_xr`) is the OVR hand rig, and
each bone origin sits on a joint of the 25-joint WebXR/OpenXR skeleton. The
retargeter (`retarget.py`) is a numpy re-implementation of the structure of
the visionOS AnyDex pipeline:

1. **Wrist.** The wrist orientation comes from a palm frame (wrist, index
   knuckle, middle knuckle), estimated identically on the tracked hand and on
   the bot's rest skeleton, so left and right hands need no mirroring
   conventions.
2. **Fingers.** Finger targets are the tracked bone *directions* rebuilt
   with the bot's own bone lengths and anchored at its knuckles. Hands of any
   size map onto the fixed-size bot (per-wearer segment scaling).
3. **Pinch.** When the wearer's fingertips are within 2–4.5 cm of the
   thumb tip, a pinch term matches the thumb-to-fingertip vectors, so pinch
   grasps close on the bot too.
4. **Solve.** A damped Gauss-Newton solve with analytic Jacobians,
   joint-limit clamping, warm start and velocity regularization computes the
   27 DoFs. It takes about 2.5 ms per hand.

The physical hand (`hand_rig.py`) ports the visionOS `HandUnit`: 5-frame
keypoint smoothing, a wrist spring (2000 N/m, 50 N·m/rad), finger joint PD
(0.5 N·m/rad, saturated at 0.6 rad), gravity compensation, a uniform hand
friction/penalty profile on both sides, and no teleporting on tracking
reacquisition.

## Tests

```bash
cd metaquest && uv run --no-project python -m pytest
```

The tests cover:

* numpy FK against the engine, and Jacobians against finite differences;
* retargeting round trips and invariance to hand size;
* every scene builds and settles;
* a scripted Quest-format grasp lifts the cube;
* per-point forces reconstruct the total contact force;
* soft-body node forces are recorded;
* the full WebSocket protocol, driven the way the headset drives it.

## Driving the server from another client

Any client that speaks the WebSocket protocol documented at the top of
`server.py` can drive the server. For example, a native OpenXR app over
Quest Link, or a replay of recorded skeletons. Send the 25 joint positions
(and optionally rotations and radii) per hand in the physics frame.
