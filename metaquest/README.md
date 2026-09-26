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
`superdex_physics/assets` (soft duck, soft cube, rope, t-shirt cloth), plus a
kitchen sponge and a dish towel. A new prefab added to `assets/prefabs/` shows
up as a scene automatically. The scenes stand in a **physical 3D kitchen**:
objects can be put in the sink, pushed off the counter and dropped on the floor.

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

All commands below run from `metaquest/` with `uv run --no-project python
...`, which uses the repository's `.venv` (macOS has no plain `python`
command; alternatively `source ../.venv/bin/activate` once per terminal and
drop the `uv run --no-project` prefix).

The PC serves everything the headset needs, including a bundled copy of
three.js, so the headset doesn't need internet access. Optional extras:
`uv pip install -e "metaquest[assets]"` for the object and room importers,
`"metaquest[video]"` for MP4 export.

**Latest SuperDex (optional).** `uv pip install superdex` installs the 1.0.0
wheels from PyPI, and everything here works with them. The current `main`
branch is about 20–40 % faster per step and fixes a t-shirt solver explosion;
to use it, build wheels from a separate clone of upstream `main` (the wheel
build refuses to run inside this branch because of the `metaquest` package)
and install them into your environment (Xcode command line tools on macOS,
Clang 17+ on Linux; roughly 15–60 minutes):

```bash
git clone https://github.com/facebookresearch/project_superdex superdex_main && cd superdex_main
uv venv -p 3.12 .venv-build
uv pip install --python .venv-build scikit-build-core cmake ninja build wheel setuptools nanobind numpy
.venv-build/bin/python tools/build_wheels.py --fast --skip-fp64 \
    --only superdex_python --only superdex_robotics --only superdex_lab --output wheelhouse
cd ../project_superdex && uv pip install --reinstall ../superdex_main/wheelhouse/*.whl
```

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

**Quit** with `Ctrl-C` in the terminal. The open recording is saved first, and
the server exits within a second. Press `Ctrl-C` again to force quit.

### In the headset

When the session starts, the virtual table (the physics origin) is placed
0.42 m in front of your eyes and 0.5 m below them, facing where you look. By
default it is the stone countertop of a kitchen, next to the sink. The
simulated floor is put at your real floor (the headset reports the counter
height). Your real hands appear as small blue joint spheres ("ghost"). The
simulated Meta XR hands follow them physically: they stop at objects instead
of passing through, so what you see matches the forces being recorded.

**Hand-size calibration.** Right after you enter VR, a panel asks you to hold
both hands open, palms down, fingers spread. After about half a second of open
hands, the PC scales the simulated hands to your size and rebuilds the scene
(the panel shows the scale and your hand length). Quest hand tracking already
fits its skeleton to your hand; the calibration compares its bone lengths with
the Meta XR hand's (palm, knuckle width, four fingers). **Calibrate** on the
panel repeats it and **Skip** keeps the current size. `--hand-scale` sets the
size without a headset, and recordings store it (`hand_scale` in the metadata).

**When a hand gets stuck.** If the simulated hand stays more than 12 cm behind
your real hand for half a second (caught on an object, or pushed deep into the
counter), it stops colliding and turns translucent. It collides again once it
has caught up and is out of the counter. Tracking glitches (a hand jumping
20 cm in one frame) can't yank the simulated hand: its targets move at most
4 m/s, 25 rad/s at the wrist and 30 rad/s per finger joint.

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
| Env ▶ | Next 3D environment (rebuilds the scene in it) |
| Colliders | Show or hide the physics colliders of the environment |
| Calibrate / Skip | Redo or skip the hand-size calibration |
| Exit VR | Leave the immersive session |

The panel also shows the scene, the simulation's real-time factor, the
headset frame rate and the recording state. It warns when no hands are
tracked, for example while you still hold the controllers.

**Desktop view:** a scene menu, **Reset**, **Record**, **Cam view**, **Env**
and **Hands**. Cam
view looks through the operator's eyes: it follows the headset live while one
is connected; otherwise it shows the default operator viewpoint. Dragging the
view returns to orbiting.

Keys: `Space` records, `R` resets, `V` toggles Cam view, `E` switches the
environment, `K` shows the physics colliders, `H` switches the hand view,
`N`/`P` changes scene, `C` toggles contacts.

### Look: environments, materials and hands

The SuperDex README videos were rendered in Unreal Engine 5 with the
unreleased *SuperDex Teleop*. The public repository has no Unreal scenes, so
this client builds its look from the assets SuperDex does ship plus CC0 assets
from [Poly Haven](https://polyhaven.com), the library SuperDex Studio already
uses for its lighting.

**Physical 3D environments.** The environment is part of the simulation. The
server builds static colliders (`workspace.py`) and sends their layout to the
client, which builds the textured kitchen from the same numbers. What you see
is what the hands and objects collide with; press `K` to overlay the
colliders. Switch with **Env** or `E` (this rebuilds the scene), or choose one
with `--environment`:

| Environment | What it is | Colliders |
| :-- | :-- | :-- |
| `kitchen_sink` (default) | Stone countertop with an undermount stainless sink to your right, gooseneck faucet, soap bottle, tiled backsplash, window, wood cabinets with handles, floor and walls, plus decorative kitchen props. It matches the SuperDex "sponge by the sink" shot | Countertop around the sink opening, a 20 cm deep basin, faucet, soap bottle, base and upper cabinets, window sill, backsplash and side walls, floor |
| `kitchen_island` | A kitchen with the work surface as a butcher-block island | Island top and body, floor: things fall off every side |
| `studio` | A table in SuperDex Studio's HDR, as a blurred backdrop | Table top and legs, floor |
| `table` | The bare infinite tabletop plane of earlier versions | One plane |

Collisions with the environment are recorded like any other contact (its
actors are named `env/...`, e.g. `env/sink_bottom`, `env/floor`). The
decorative props on the far counters are display only.

**Poly Haven asset pack.** On first start, the server downloads a CC0 pack
into `~/.superdex_quest_teleop/assets` in the background, and connected
headsets switch to it as soon as it's ready. You can also fetch it yourself:

```bash
uv run --no-project python -m superdex_quest_teleop.environment              # 1k textures (Quest)
uv run --no-project python -m superdex_quest_teleop.environment --resolution 2k
```

It contains:

* **Kitchen HDRIs** for image-based lighting and reflections: `blinds` (a
  morning kitchen), `kiara_interior` (a kitchen lounge) and `lebombo`.
* **PBR material sets** (albedo, OpenGL normal and roughness maps), tiled at
  real-world scale:

  | Slot | Assets |
  | :-- | :-- |
  | Countertop | `marble_01`, `granite_tile` |
  | Cabinets | `kitchen_wood` |
  | Backsplash | `long_white_tiles`, `interior_tiles` |
  | Floor | `laminate_floor` |
  | Walls | `white_plaster_02` |
  | Table top | `oak_wood_planks` |
  | Brushed steel | `metal_plate_02` |
* **glTF props** (for example `food_lime_01`, `wooden_spoon`, and matching
  food and kitchen models) placed on the counters.

Curated asset IDs are tried first, then a keyword search over Poly Haven's
live catalog. Without the pack (offline, or `--no-download`), procedural
materials stand in.

**Objects.** Every prefab is drawn with its upstream textured PBR model
(`assets/prefabs/*/render/*.glb`: the wooden box and blocks, pegboards,
printed paper cups, shape sorter, chain, sphere). The kitchen sponge is a
yellow foam block with a green scouring layer whose pores stay attached as it
deforms. All objects and hands cast soft shadows on the counter.

**Hands.** As in the SuperDex Teleop renders, each hand is a single smooth
skinned hand that ends in a rounded wrist, with no forearm. It uses the WebXR
`generic-hand` model (MIT) with light skin, subtle normal detail and a soft
sheen. Every frame, the PC turns the *simulated* Meta XR hand's link poses
into the 25 joint poses the mesh is rigged to, so the skin shows exactly where
physics put the fingers. Press **H** to see the Meta XR robot-hand meshes that
actually collide.

**BEDLAM hands.** `superdex_quest_teleop.bedlam_hands` builds photoreal hands
from a SMPL-X model and a BEDLAM (or BEDLAM 2.0) scanned skin texture. It:

* cuts both hands out of the SMPL-X body (optionally with shape `--betas`);
* closes the wrist with a dome;
* re-skins the hands to the 25 WebXR joints;
* crops the skin texture to the hands' UV region;
* writes `left.glb` and `right.glb`.

Run it where BEDLAM lives (e.g. your cluster):

```bash
uv run --no-project python -m superdex_quest_teleop.bedlam_hands --smplx /path/to/smplx/SMPLX_NEUTRAL.npz \
    --bedlam-root /CT/datasets10/static00/BEDLAM --list            # list skin textures
uv run --no-project python -m superdex_quest_teleop.bedlam_hands --smplx /path/to/smplx/SMPLX_NEUTRAL.npz \
    --bedlam-root /CT/datasets26/static00/BEDLAM_2 --texture-match <name> --out ~/bedlam_hands
uv run --no-project python -m superdex_quest_teleop --hand-models ~/bedlam_hands
```

Notes:

* The SMPL-X `.npz` must contain UVs (`vt`/`ft`), or pass `--uv-obj`
  pointing at the SMPL-X UV template.
* Use `--texture` to choose an exact albedo file.
* SMPL-X and BEDLAM are licensed for non-commercial research, and so are the
  generated files: keep them out of public repositories.
* Any other rigged hand whose bones use the 25 WebXR joint names works with
  `--hand-models` too.

**Your own kitchen.** Pass `--environment` a 360° photo (`.jpg`/`.png`, taken
from where you stand), an `.hdr`, or a `.glb`/`.gltf` model. The model's
origin goes on the floor under the table center, with -Z facing away from
you. On Quest 3, **Enter passthrough** shows your real kitchen.

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
| `--scene ID` | `kitchen_sponge` | Initial scene (`--list-scenes`) |
| `--out DIR` | `recordings` | Where episodes are written |
| `--contacts hand\|all` | `hand` | `hand`: every contact involving a hand link. `all`: also object/object and object/table contacts |
| `--min-contact-force N` | `0` | Drop contact points with smaller force. `0` also keeps zero-force near-contact samples |
| `--keep-self-contacts` | off | Also record contacts between links of the same hand |
| `--hand-mesh lowpoly\|highpoly` | `lowpoly` | Meta XR hand collision mesh |
| `--record` | off | Start recording immediately |
| `--synthetic` | off | Scripted right hand instead of the headset |
| `--http-port N` / `--https-port N` | `8080` / `8443` | Listening ports |
| `--no-https` | off | Serve plain HTTP only |
| `--environment NAME\|FILE` | `kitchen_sink` | `kitchen_sink`, `kitchen_island`, `studio`, `table`, or a backdrop file (`.hdr`, 360° photo, `.glb`) shown around the island |
| `--no-download` | off | Don't fetch the Poly Haven asset pack |
| `--asset-dir DIR` / `--asset-resolution` | `~/.superdex_quest_teleop/assets` / `1k` | Asset pack location and texture resolution |
| `--hand-models DIR` | WebXR generic hand | Rigged, textured `left.glb`/`right.glb` (e.g. from `bedlam_hands`) |
| `--hand-scale S` | `1` | Size of the simulated hands (normally set by the in-headset calibration) |
| `--time-budget F` | `0.8` | Cap each step's solve at this fraction of the time step, so heavy scenes stay real time. `0`: solve to the engine's tolerances even if slower than real time |
| `--grip-strength F` | `1` | Finger torque cap multiplier (1 = 0.64 N·m per joint). Raise it if heavy objects slip out of a firm grasp |
| `--threads N` | `-1` | SuperDex worker threads |

## Scenes

The work surface is at y = 0 (the countertop of the chosen environment). Z-up
prefabs are rotated into the Y-up physics frame (as in the visionOS prefab
gallery) and centered on it using their measured bounds.

| id | Kind | Content |
| :-- | :-- | :-- |
| `kitchen_sponge` | soft + rigid | A soft kitchen sponge and a paper cup by the sink (default) |
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
| `cloth` | shell | 45 cm t-shirt with self contact (1/30 s step, one Newton iteration) |
| `dish_towel` | shell | 30 cm cotton towel (light: 4× real time) |
| `medley` | mixed | Sphere, cup, block, peg, cube and a soft duck |

To add a scene, drop a prefab into `assets/prefabs/<name>/`, or add a
`SceneSpec` to `superdex_quest_teleop/scenes.py`.

The loop runs at the scene's fixed step and paces itself to wall-clock time.
If a step takes longer than its duration, the simulation runs in slow motion
rather than skipping steps. The HUD shows the real-time factor (RTF).

Heavy scenes stay real time with the solve time budget (`--time-budget`,
default 0.8 of the step): a step's Newton iterations stop when the budget is
spent. Real-time factors measured on a 4-core cloud VM with one hand grasping
(1.0.0 wheel / current `main`):

* **Real time and faster:** `sphere` (3.7× / 5.3×), `kitchen_sponge`
  (2.4× / 3.2×), `dish_towel` (2.4× / 4×), `cube`, `paper_cup`,
  `nine_hole_peg_test`, `functional_dexterity_test`, `medley` (1.05× / 1.4×).
* **About real time:** `cloth` (0.95× / 1.13×; it was 0.15–0.26× before the
  t-shirt used one Newton iteration and streamed its 3.6k-node simulation mesh
  instead of its 57k-vertex subdivided visual mesh), `soft_cube`,
  `free_rope`, `soft_duck`.
* **Slower than real time:** `paper_cup_pyramid`, `shape_box`, `chain`.

More cores help. Recording costs a few microseconds per contact point, so
contact-dense moments slow the loop down while recording.

## Objects with measured weight and realistic friction

Force data is only as good as the objects' mass and friction. `objects.py`
converts simulation assets with measured or identified physical parameters
into SuperDex prefabs (`.mochi_prefab` + `.mochi.h5` collision shape +
textured render GLB) in `~/.superdex_quest_teleop/objects`, and every
converted object becomes a scene.

```bash
uv pip install trimesh scipy scikit-image rtree fast-simplification coacd huggingface_hub
uv run --no-project python -m superdex_quest_teleop.objects fetch ycb          # 17 YCB objects (about 2 min)
uv run --no-project python -m superdex_quest_teleop.objects fetch real2sim     # Scalable Real2Sim benchmark objects
uv run --no-project python -m superdex_quest_teleop.objects fetch scenesmith --limit 50 [--match mug]
uv run --no-project python -m superdex_quest_teleop.objects convert my_model.sdf --set mine [--material plastic]
uv run --no-project python -m superdex_quest_teleop.objects list
```

Then pick `obj_ycb_011_banana` (one object) or `objects_ycb_1`… (six
objects on the counter) from the scene menu.

| Set | Objects | Mass | Geometry |
| :-- | :-- | :-- | :-- |
| `ycb` | The 16 YCB-Video objects (cans, boxes, mustard bottle, banana, pear, strawberry, drill, hammer, scissors, clamp, tennis ball, foam brick, …) and the sugar box | Measured, from the YCB paper (Calli et al. 2015); model files that disagree are corrected (the foam brick file says 28 g; YCB measured 59 g) | Textured scans (pybullet-object-models, Drake models) |
| `real2sim` | Scalable Real2Sim benchmark objects (Pfaff et al. 2025) | Identified from robot joint torques, with center of mass and inertia | Photometric reconstructions |
| `scenesmith` | SceneSmith SAM-3D objects (Pfaff et al. 2026) | Estimated by a vision-language model: plausible, not measured | Generated meshes |
| any `.sdf`/`.urdf` | Drake/Gazebo/pybullet models | From `<inertial>` | From the file |

**Friction.** No open object dataset measures friction (simulator URDFs carry
placeholders: every pybullet YCB object says 0.8; SceneSmith writes
same-material values such as steel on steel 0.74). Since hand contacts are
what gets recorded, each object gets the measured friction of dry fingertip
skin against its material (central values from the skin tribology
literature: Derler & Gerhardt 2012, Tomlinson et al. 2007, Skedung et al.
2010):

| Material | Fingertip μ | | Material | Fingertip μ |
| :-- | :-- | :-- | :-- | :-- |
| cardboard | 0.57 | | glass, ceramic | 0.45 |
| paper | 0.55 | | fabric | 0.45 |
| plastic | 0.50 | | fruit peel | 0.55 |
| metal | 0.40 | | foam | 0.90 |
| wood | 0.50 | | rubber | 1.20 |

The material comes from the object's name (or SceneSmith's friction value, or
`--material`); `--mu` sets the fingertip friction directly. SuperDex combines
two actors' coefficients by their geometric mean and the hands use 1.0, so an
object's coefficient is μ²: finger contacts then have exactly the table's μ,
and the object slides on the countertop (0.6) at 0.77 μ (cardboard on stone
0.44). Skin friction varies about ±40 % with moisture.

**Collision shapes.** SuperDex samples contact on both surfaces, so the
collision shape follows the real surface. Primitives are kept where they fit
the scan within 1.2 cm (the sugar box). Otherwise the scanned mesh is
re-meshed through its signed distance field into a closed mesh of about 3000
triangles (0.2–0.8 mm from the scan, concave shapes such as a banana or a
clamp preserved), or kept as is for thin shells (plates, scissors). Contact
uses a 1 mm smoothing band, so objects rest within a millimetre of the
counter. Each object's `physics.json` records its mass, inertia, material,
friction, collision shape and source.

## Rooms (Sketchfab or any glTF) as physical environments

Beyond the built-in kitchens, any room model can be the environment: the
Sketchfab rooms below, or any glTF/GLB/zip. `rooms.py` imports a room into
`~/.superdex_quest_teleop/rooms` and makes it physical around a table or desk:

```bash
export SKETCHFAB_API_TOKEN=...   # sketchfab.com -> Settings -> Password & API
uv run --no-project python -m superdex_quest_teleop.rooms add retro_apartment   # "Modern Retro Apartment"
uv run --no-project python -m superdex_quest_teleop.rooms add sherlock_221b     # "221B Baker Street - Sherlock - Archilogic"
uv run --no-project python -m superdex_quest_teleop.rooms add living_room       # "Living Room"
uv run --no-project python -m superdex_quest_teleop.rooms add ~/Downloads/some_room.zip --name den   # a downloaded glTF
uv run --no-project python -m superdex_quest_teleop.rooms surfaces sherlock_221b            # work-surface candidates
uv run --no-project python -m superdex_quest_teleop.rooms add sherlock_221b --surface 2     # use another one
uv run --no-project python -m superdex_quest_teleop --environment sherlock_221b --scene objects_ycb_1
```

The presets also import on first use with `--environment NAME` when
`SKETCHFAB_API_TOKEN` is set. **Env** cycles through the built-in and
imported environments.

The import:

1. **Units and floor.** It finds the floor (the lowest large upward-facing
   area) and the ceiling, and picks the unit (m, cm, mm, inch, ft) that
   makes the room 2.2–4.5 m high.
2. **Work surface.** It lists upward-facing areas 0.35–1.15 m above the
   floor that are at least 45 × 30 cm (tables, desks, counters), preferring
   0.7–0.95 m. The physics origin goes on the chosen one, its long side
   along X, and you stand on its freer side.
3. **Colliders.** Everything within 1.4 m becomes static convex colliders:
   a CoACD decomposition near the surface, convex hulls further out, wall
   slabs behind the walls, an analytic floor, and an exact slab for the
   work surface. `K` shows them over the model. Props that are part of the
   room model (a vase on the desk) are static obstacles; the objects you
   manipulate come from the scene.

In VR a room keeps its real proportions: the work surface stays at its
modeled height above your real floor (0.75 m for a desk), instead of 0.5 m
below your eyes as in the kitchens. **Table ▲/▼** still adjusts it.

Each room's license and author are recorded in its `room.json` and shown
with the scene description. Sketchfab's CC Attribution models require that
credit; check the license before sharing recordings or videos. Very detailed
models can be heavy for the Quest browser: prefer models under about 500k
triangles.

## Recorded data

Each episode is one HDF5 file, `<out>/<scene>_<timestamp>.h5`. With T steps,
N actors (objects, the environment colliders and 19 links per hand) and K
contact points in total:

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
Quest joint names, the recording options, the environment with its collider
layout, the counter height and the hand scale.

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

### Viewing an episode

**Replay in 3D.** Play an episode back in the same viewer, in a desktop
browser or in the headset:

```bash
cd metaquest
uv run --no-project python -m superdex_quest_teleop.replay ../recordings/kitchen_sponge_20260925_101500.h5
```

Then open `http://localhost:8080/`. You see the objects, the deforming bodies,
the hands and the recorded contact points with their force vectors (`C`),
with a timeline: play/pause (`Space`), scrub, step (`←`/`→`, with `Shift` 10
steps) and change the speed. Nothing is simulated: the scene is rebuilt only
for its meshes, and everything that moves comes from the file.

**Export an MP4.** Click **Export MP4** in the replay bar to render the
episode as you see it (current view and playback speed, 30 fps). The file is
written next to the episode and the page offers a download link. From the
command line, without opening a viewer:

```bash
uv pip install playwright imageio-ffmpeg        # once (or: pip install -e "metaquest[video]")
uv run --no-project python -m playwright install chromium   # once; or pass --browser /path/to/chrome
uv run --no-project python -m superdex_quest_teleop.replay ../recordings/kitchen_sponge_20260925_101500.h5 --mp4
```

This renders the episode in a headless browser with the same kitchen, hands
and contact forces, and writes `kitchen_sponge_20260925_101500.mp4` (H.264)
next to it. Options:

| Flag | Default | Meaning |
| :-- | :-- | :-- |
| `--mp4 [OUT.mp4]` | episode name | Render to this file and exit |
| `--size WxH` | `1920x1080` | Video size |
| `--fps N` | `30` | Frame rate |
| `--speed S` | `1` | Playback speed (`0.25`: 4× slow motion) |
| `--start S` / `--end S` | whole episode | Time range [s] |
| `--view orbit\|cam` | `orbit` | A fixed camera over the counter, or through the operator's eyes (recorded headset pose) |
| `--camera PX,PY,PZ,TX,TY,TZ` | `0,1.55,0.75,0.05,0.9,-0.1` | Orbit camera position and target, room coordinates (counter top at y = 0.9 m) |
| `--no-contacts` | off | Hide contact points and force vectors |
| `--no-overlay` | off | No scene name / time caption |
| `--browser PATH` | Playwright's Chromium, then Google Chrome | Browser used for rendering |
| `-v` | off | Print the page's console |

Every video frame is rendered at its exact recorded step, so the video is
smooth however slow the rendering is. It is fast with a GPU; on a machine
without one (e.g. a headless Linux server), WebGL falls back to software
rendering at about one 1080p frame per second.

**In Python.** Episodes are plain HDF5:

```python
import h5py, json
with h5py.File("recordings/kitchen_sponge_20260925_101500.h5") as f:
    meta = json.loads(f.attrs["metadata"])
    names = [n.decode() for n in f["actors/name"][:]]
    pose = f["actors/pose"][:]                       # (T, N, 7)
    t = 100                                          # contacts of step 100:
    rows = slice(*f["contacts/step_offset"][t:t + 2])
    force = f["contacts/force"][rows]                # (k, 3) N
    point = f["contacts/pos_owner"][rows]            # (k, 3) m
    owner = [names[i] for i in f["contacts/owner"][rows]]
```

**Other tools.** Any HDF5 viewer opens the files, for example
[HDFView](https://www.hdfgroup.org/downloads/hdfview/),
[myHDF5](https://myhdf5.hdfgroup.org/) in the browser, the VS Code "H5Web"
extension, or `h5ls -r file.h5` on the command line.

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
keypoint smoothing, a wrist spring (2000 N/m, 50 N·m/rad), gravity
compensation, a uniform hand penalty profile on both sides, and no teleporting
on tracking reacquisition. It differs from the visionOS values in three ways:

* **Finger joints** are stiffer: 0.8 N·m/rad, saturated at 0.8 rad (up to
  0.64 N·m per joint, twice the visionOS grip). Much stiffer fingers squeeze
  small objects out of an open-loop grasp.
* **Friction:** the hand's Coulomb coefficient is 1.0, giving 0.7 against
  most objects (the pair uses the geometric mean). The visionOS 2.0 made
  objects cling to the fingers.
* **Targets** are rate limited, and a stuck hand passes through objects until
  it catches up (see *When a hand gets stuck*).

## Tests

```bash
cd metaquest && uv run --no-project python -m pytest
```

The tests cover:

* numpy FK against the engine, and Jacobians against finite differences;
* retargeting round trips and invariance to hand size;
* every scene builds and settles;
* a scripted Quest-format grasp lifts the cube;
* objects land in the sink, on the counter and on the floor, which follows the
  counter height; the island and studio table have edges;
* hand-size calibration and the scaled hand bot;
* SDFormat/URDF objects become prefabs with their mass, material friction and
  closed collision shape; SceneSmith friction values map back to materials;
* a room imports with the right units, work surface and facing, and
  objects land on the desk, on the floor and against the wall;
* the unstick logic, and rate limiting of tracking jumps;
* an episode replays frame-exactly;
* per-point forces reconstruct the total contact force;
* soft-body node forces are recorded;
* the full WebSocket protocol, driven the way the headset drives it.

## Driving the server from another client

Any client that speaks the WebSocket protocol documented at the top of
`server.py` can drive the server. For example, a native OpenXR app over
Quest Link, or a replay of recorded skeletons. Send the 25 joint positions
(and optionally rotations and radii) per hand in the physics frame.
