// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// SuperDex Quest teleop client. In the Quest browser it runs a WebXR session
// with hand tracking, streams both 25-joint hand skeletons (in the physics
// frame) to the PC, and renders the simulated scene the PC streams back. On
// a desktop browser the same page is a spectator view with the same
// controls, including a "Cam view" that looks through the headset.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { SkinnedHand } from './hands.js';
import { buildKitchen, makeIsland } from './kitchen.js';
import { makeSkinMaterial } from './skin.js';

const CONFIG = window.SUPERDEX || {};
const JOINTS = 25;
const TIP_INDEX = 9;              // index-finger-tip
const SEND_PERIOD_MS = 1000 / 72; // hand message rate cap
const TABLE_BELOW_HEAD = 0.5;     // [m] table height below the eyes on recenter
const TABLE_AHEAD = 0.42;         // [m] table center in front of the eyes
const DESKTOP_TABLE_HEIGHT = 0.9; // [m] kitchen counter height
// Where the operator's eyes are, in the physics frame, right after a recenter.
const DEFAULT_HEAD = new THREE.Vector3(0, TABLE_BELOW_HEAD, TABLE_AHEAD);

// --------------------------------------------------------------------------
// Messages
// --------------------------------------------------------------------------

function banner(html, kind = 'error') {
  const b = document.getElementById('banner');
  document.getElementById('banner-text').innerHTML = html;
  b.className = kind === 'info' ? 'info' : '';
  b.style.display = 'block';
}
function hideBanner() { document.getElementById('banner').style.display = 'none'; }

let flashText = '', flashUntil = 0;
function flash(text) { flashText = text; flashUntil = performance.now() + 5000; }

function isLocalHost() {
  return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(location.hostname);
}
function secureContextHelp() {
  const https = CONFIG.httpsPort ? `https://${location.hostname}:${CONFIG.httpsPort}/` : null;
  const http = CONFIG.httpPort || location.port || 8080;
  return 'WebXR (and hand tracking) only works on secure pages. ' +
    (https ? `On the Quest open <a href="${https}">${https}</a> and accept the certificate warning once ` +
      '(<i>Advanced → Proceed</i>), or ' : '') +
    `connect by USB: run <code>adb reverse tcp:${http} tcp:${http}</code> on the PC and open ` +
    `<code>http://localhost:${http}/</code> in the Quest browser.`;
}

// --------------------------------------------------------------------------
// Renderer and scene graph
// --------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.8;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const background = new THREE.Color(0xb9c6d2);
scene.background = background;
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
let photoEnvironment = false;
// Image-based lighting from SuperDex Studio's own HDR (Poly Haven, CC0),
// replaced by the kitchen HDRI's light once that loads.
if (CONFIG.ibl) {
  new RGBELoader().load(CONFIG.ibl, (hdr) => {
    if (photoEnvironment) return;
    hdr.mapping = THREE.EquirectangularReflectionMapping;
    scene.environment = pmrem.fromEquirectangular(hdr).texture;
    hdr.dispose();
  });
}
const hemi = new THREE.HemisphereLight(0xfff6ea, 0x8a7a68, 0.35);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 0.9);
sun.position.set(-1.0, 2.6, -1.2);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 60);
camera.position.set(0, 1.55, 0.85);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, DESKTOP_TABLE_HEIGHT, -0.05);
// ?cam=px,py,pz,tx,ty,tz: start the desktop view at a given camera/target.
const camParam = new URLSearchParams(location.search).get('cam');
if (camParam) {
  const v = camParam.split(',').map(Number);
  if (v.length === 6 && v.every(Number.isFinite)) {
    camera.position.set(v[0], v[1], v[2]);
    controls.target.set(v[3], v[4], v[5]);
  }
}
controls.update();

// Physics frame (Y-up, meters, table top at y = 0). Its placement in the room
// is the workspace anchor; everything simulated lives under it.
const workspace = new THREE.Group();
workspace.matrixAutoUpdate = false;
scene.add(workspace);

const kitchen = buildKitchen();
kitchen.group.matrixAutoUpdate = false;
scene.add(kitchen.group);
const island = makeIsland();
workspace.add(island.group);

let anchor = new THREE.Matrix4();
function setAnchor(m) {
  anchor = m.clone();
  workspace.matrix.copy(anchor);
  workspace.matrixWorldNeedsUpdate = true;
  // The kitchen stands on the floor under the table, turned with it.
  const p = new THREE.Vector3().setFromMatrixPosition(anchor);
  const yaw = Math.atan2(anchor.elements[8], anchor.elements[10]);
  kitchen.group.matrix.makeRotationY(yaw).setPosition(p.x, 0, p.z);
  kitchen.group.matrixWorldNeedsUpdate = true;
  island.setHeight(p.y);
}
setAnchor(new THREE.Matrix4().makeTranslation(0, DESKTOP_TABLE_HEIGHT, 0));

// Backdrop: a photographed kitchen (HDRI, auto-fetched from Poly Haven by
// the PC), a 360 photo or a model of the user's own kitchen. The HDRI also
// lights the scene, so objects and hands match the room they appear in.
let environmentDome = null;
function applyEnvironment(env) {
  if (!env || (environmentDome && environmentDome.userData.url === env.url)) return;
  const place = (object) => {
    if (environmentDome) kitchen.group.remove(environmentDome);
    environmentDome = object;
    environmentDome.userData.url = env.url;
    kitchen.room.visible = false;
    kitchen.group.add(object);
    flash(`environment: ${env.name || env.url}`);
  };
  const dome = (texture) => {
    const sphere = new THREE.SphereGeometry(25, 64, 32);
    sphere.scale(-1, 1, 1);  // faces inward, image not mirrored
    const mesh = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({ map: texture }));
    mesh.position.y = 1.5;   // roughly the camera height of room HDRIs
    return mesh;
  };
  const fail = () => banner(`Could not load the environment ${env.url}.`);
  if (env.kind === 'hdr') {
    new RGBELoader().load(env.url, (hdr) => {
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      scene.environment = pmrem.fromEquirectangular(hdr).texture;
      photoEnvironment = true;
      hemi.intensity = 0.1;
      sun.intensity = 0.35;
      place(dome(hdr));
    }, undefined, fail);
  } else if (env.kind === 'panorama') {
    new THREE.TextureLoader().load(env.url, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      place(dome(texture));
    }, undefined, fail);
  } else {
    new GLTFLoader().load(env.url, (gltf) => place(gltf.scene), undefined, fail);
  }
}
applyEnvironment(CONFIG.environment);

const actorRoot = new THREE.Group();
workspace.add(actorRoot);

// Contacts: points colored by force, plus force vectors.
const MAX_CONTACTS = 256;
const contactDots = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.0025, 8, 6),
  new THREE.MeshBasicMaterial({ color: 0xffffff }), MAX_CONTACTS);
contactDots.count = 0;
contactDots.frustumCulled = false;
workspace.add(contactDots);
const forceLines = new THREE.LineSegments(
  new THREE.BufferGeometry().setAttribute(
    'position', new THREE.BufferAttribute(new Float32Array(MAX_CONTACTS * 6), 3)),
  new THREE.LineBasicMaterial({ color: 0xffd166 }));
forceLines.frustumCulled = false;
workspace.add(forceLines);
let showContacts = true;

// Raw tracked joints (the operator's real hands) as small ghost spheres.
const ghost = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.004, 8, 6),
  new THREE.MeshBasicMaterial({ color: 0x7fd3ff, transparent: true, opacity: 0.5 }), 2 * JOINTS);
ghost.count = 0;
ghost.frustumCulled = false;
scene.add(ghost);
let showGhost = true;

// --------------------------------------------------------------------------
// Scene geometry from the server
// --------------------------------------------------------------------------

let actors = [];          // index -> {object, deformable}
let sceneList = [];
let currentScene = null;
let lastHeader = null;
let statusInfo = { recording: false };

const skin = makeSkinMaterial();
const gltfLoader = new GLTFLoader();
const renderMeshCache = new Map();  // url -> Promise<BufferGeometry>

const renderSceneCache = new Map();  // url -> Promise<Object3D>
function loadRenderScene(url) {
  if (!renderSceneCache.has(url)) {
    renderSceneCache.set(url, new Promise((resolve, reject) => {
      gltfLoader.load(url, (gltf) => resolve(gltf.scene), undefined, reject);
    }));
  }
  return renderSceneCache.get(url).then((s) => s.clone(true));
}

// Continuous skinned hands posed from the simulated Meta XR hands.
const HAND_MODELS = CONFIG.handModels || {
  left: '/vendor/webxr-input-profiles/generic-hand/left.glb',
  right: '/vendor/webxr-input-profiles/generic-hand/right.glb',
};
const skinnedHands = {};
for (const side of ['left', 'right']) {
  if (!HAND_MODELS[side]) continue;
  const hand = new SkinnedHand(HAND_MODELS[side]);
  skinnedHands[side] = hand;
  workspace.add(hand.group);
  hand.ready.catch((err) => banner(`Could not load the ${side} hand model: ${err.message || err}`));
}
let handView = 'skin';  // 'skin': skinned hands; 'robot': Meta XR link meshes
function setHandView(view) {
  handView = view;
  for (const a of actors) if (a && a.hand && a.object) a.object.visible = view === 'robot';
  for (const h of Object.values(skinnedHands)) h.group.visible = view === 'skin' && !!h.bones;
  const b = document.getElementById('handview');
  if (b) b.textContent = view === 'skin' ? 'Hands: skin' : 'Hands: robot';
}

function loadRenderGeometry(url) {
  if (!renderMeshCache.has(url)) {
    renderMeshCache.set(url, new Promise((resolve, reject) => {
      gltfLoader.load(url, (gltf) => {
        let geometry = null;
        gltf.scene.traverse((o) => { if (!geometry && o.isMesh) geometry = o.geometry; });
        geometry ? resolve(geometry) : reject(new Error('no mesh'));
      }, undefined, reject);
    }));
  }
  return renderMeshCache.get(url);
}

// Display-only forearm behind the wrist link (the hand model ends at the
// wrist). Right wrist frame: fingers along +Y, palm width along X, back of
// the hand toward -Z; the left asset is mirrored (fingers along -Y).
function makeForearm(side) {
  const length = 0.16;
  const geometry = new THREE.CylinderGeometry(0.92, 1.15, length, 28);
  geometry.translate(0, -length / 2 + 0.02, 0);
  const forearm = new THREE.Mesh(geometry, skin);
  forearm.scale.set(0.029, 1, 0.021);
  forearm.position.z = -0.011;
  const holder = new THREE.Group();
  holder.add(forearm);
  if (side === 'left') holder.rotation.x = Math.PI;
  return holder;
}

const PALETTE = [0xe4572e, 0x29a19c, 0xf3a712, 0x6c8ead, 0xa8c686, 0xd183c9, 0x4f86c6, 0xf08a4b];
function colorFor(name) {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  if (/red/i.test(name)) return 0xd64541;
  if (/green/i.test(name)) return 0x4caf50;
  if (/blue/i.test(name)) return 0x3f7fd6;
  if (/yellow/i.test(name)) return 0xf2c230;
  if (/duck/i.test(name)) return 0xf5c518;
  if (/cup/i.test(name)) return 0xf1eee6;
  return PALETTE[h % PALETTE.length];
}

function buildGeometry(msg) {
  actorRoot.clear();
  actors = [];
  currentScene = msg.scene;
  document.getElementById('desc').textContent = msg.scene.description || '';
  document.getElementById('scene').value = msg.scene.id;
  for (const a of msg.actors) {
    const hand = a.hand !== '';
    const entry = { object: null, deformable: a.deformable, hand };
    if (a.mesh === 'surface' || a.mesh === 'visual') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(a.positions), 3));
      g.setIndex(a.indices);
      g.computeVertexNormals();
      const material = hand ? skin : new THREE.MeshStandardMaterial({
        color: a.static ? 0x9aa0a8 : colorFor(a.name),
        roughness: 0.45,
        metalness: 0.0,
        side: a.deformable ? THREE.DoubleSide : THREE.FrontSide,
        flatShading: !a.deformable,
      });
      const mesh = new THREE.Mesh(g, material);
      mesh.frustumCulled = false;
      if (a.deformable) {
        mesh.matrixAutoUpdate = false;  // vertices stream in world coordinates
        actorRoot.add(mesh);
        entry.object = mesh;
      } else {
        const holder = new THREE.Group();
        holder.add(mesh);
        actorRoot.add(holder);
        entry.object = holder;
        if (hand && /bone_00_wrist_root$/.test(a.name)) holder.add(makeForearm(a.hand));
        if (!hand && a.render) {
          // Upstream textured PBR model of this prefab actor.
          loadRenderScene(a.render.url).then((model) => {
            model.quaternion.fromArray(a.render.rotation);
            model.traverse((o) => { if (o.isMesh) o.frustumCulled = false; });
            holder.remove(mesh);
            holder.add(model);
          }).catch(() => {});
        }
        if (hand) holder.visible = handView === 'robot';
        if (hand && a.render) {
          // Swap the coarse collision mesh for the smooth render mesh.
          loadRenderGeometry(a.render.url).then((geometry) => {
            const smooth = new THREE.Mesh(geometry, skin);
            smooth.quaternion.fromArray(a.render.rotation);
            smooth.frustumCulled = false;
            holder.remove(mesh);
            holder.add(smooth);
          }).catch(() => {});
        }
      }
    }
    actors[a.index] = entry;
  }
}

function applyFrame(buffer) {
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  let offset = 4 + headerLength;
  offset += (4 - (offset % 4)) % 4;
  const f = new Float32Array(buffer, offset);
  lastHeader = header;
  let k = 0;
  for (let i = 0; i < header.num_actors; i++, k += 7) {
    const a = actors[i];
    if (!a || !a.object || a.deformable) continue;
    a.object.position.set(f[k], f[k + 1], f[k + 2]);
    a.object.quaternion.set(f[k + 3], f[k + 4], f[k + 5], f[k + 6]);
  }
  for (const [index, count] of header.deformables) {
    const a = actors[index];
    if (a && a.object) {
      const attr = a.object.geometry.getAttribute('position');
      if (attr.count === count) {
        attr.array.set(f.subarray(k, k + 3 * count));
        attr.needsUpdate = true;
        if ((header.step & 3) === 0) a.object.geometry.computeVertexNormals();
      }
    }
    k += 3 * count;
  }
  const n = Math.min(header.num_contacts, MAX_CONTACTS);
  const points = f.subarray(k, k + 3 * n);
  const forces = f.subarray(k + 3 * header.num_contacts, k + 3 * header.num_contacts + 3 * n);
  updateContacts(points, forces, n);
  k += 6 * header.num_contacts;
  for (const side of header.hand_joints || []) {
    const hand = skinnedHands[side];
    if (hand && handView === 'skin') hand.setJoints(f.subarray(k, k + 175));
    k += 175;
  }
}

const _m = new THREE.Matrix4();
const _c = new THREE.Color();
function updateContacts(points, forces, n) {
  contactDots.count = showContacts ? n : 0;
  const lines = forceLines.geometry.getAttribute('position');
  for (let i = 0; i < n; i++) {
    const px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
    const fx = forces[3 * i], fy = forces[3 * i + 1], fz = forces[3 * i + 2];
    const mag = Math.hypot(fx, fy, fz);
    _m.makeTranslation(px, py, pz);
    contactDots.setMatrixAt(i, _m);
    const t = Math.min(1, mag / 0.5);
    _c.setRGB(0.2 + 0.8 * t, 0.6 * (1 - t) + 0.2, 1 - t);
    contactDots.setColorAt(i, _c);
    const s = mag > 1e-9 ? Math.min(0.04, 0.02 * mag) / mag : 0;
    lines.array.set([px, py, pz, px + fx * s, py + fy * s, pz + fz * s], 6 * i);
  }
  contactDots.instanceMatrix.needsUpdate = true;
  if (contactDots.instanceColor) contactDots.instanceColor.needsUpdate = true;
  forceLines.geometry.setDrawRange(0, showContacts ? 2 * n : 0);
  lines.needsUpdate = true;
}

// --------------------------------------------------------------------------
// Connection
// --------------------------------------------------------------------------

let ws = null;
let connected = false;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    connected = true;
    ws.send(JSON.stringify({ type: 'hello', role: xrSession ? 'headset' : 'viewer' }));
  };
  ws.onmessage = (event) => {
    if (typeof event.data !== 'string') { applyFrame(event.data); return; }
    const msg = JSON.parse(event.data);
    if (msg.type === 'hello') {
      sceneList = msg.scenes;
      const sel = document.getElementById('scene');
      sel.innerHTML = '';
      for (const s of sceneList) {
        const o = document.createElement('option');
        o.value = s.id; o.textContent = `${s.name}  [${s.category}]`;
        sel.appendChild(o);
      }
      if (currentScene) sel.value = currentScene.id;
      applyEnvironment(msg.environment);
    } else if (msg.type === 'environment') {
      applyEnvironment(msg.environment);
    } else if (msg.type === 'geometry') {
      buildGeometry(msg);
    } else if (msg.type === 'status') {
      if (msg.error) flash(`error: ${msg.error}`);
      if (msg.message) flash(msg.message);
      if (msg.saved) flash(`saved ${msg.saved_steps} steps -> ${msg.saved}`);
      if ('recording' in msg) statusInfo = msg;
      const rec = document.getElementById('rec');
      rec.classList.toggle('on', !!statusInfo.recording);
      rec.textContent = statusInfo.recording ? '■ Stop' : '● Record';
    }
  };
  ws.onclose = () => { connected = false; flash('disconnected from the PC, retrying…'); setTimeout(connect, 1000); };
}
connect();

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}
function command(cmd, extra = {}) { send({ type: 'cmd', cmd, ...extra }); }

// --------------------------------------------------------------------------
// Desktop controls and Cam view
// --------------------------------------------------------------------------

// Cam view: look through the operator's eyes (the streamed headset pose, or
// the default operator position when no headset is connected).
let camView = false;
const _head = new THREE.Matrix4();
const _target = new THREE.Vector3();
function setCamView(on) {
  camView = on;
  controls.enabled = !on;
  document.getElementById('camview').classList.toggle('on', on);
  camera.fov = on ? 80 : 55;
  camera.updateProjectionMatrix();
  if (!on) {
    // Keep looking where the headset looked, orbiting about the table.
    controls.target.setFromMatrixPosition(anchor);
    controls.update();
  }
}
function updateCamView() {
  const head = lastHeader && lastHeader.head;
  if (head) {
    _head.compose(new THREE.Vector3(head[0], head[1], head[2]),
      new THREE.Quaternion(head[3], head[4], head[5], head[6]), new THREE.Vector3(1, 1, 1));
  } else {
    _head.lookAt(DEFAULT_HEAD, _target.set(0, 0, -0.05), new THREE.Vector3(0, 1, 0));
    _head.setPosition(DEFAULT_HEAD);
  }
  _head.premultiply(anchor);  // physics -> room
  _head.decompose(camera.position, camera.quaternion, camera.scale);
}
controls.addEventListener('start', () => { if (camView) setCamView(false); });

document.getElementById('scene').onchange = (e) => command('load_scene', { scene: e.target.value });
document.getElementById('reset').onclick = () => command('reset');
document.getElementById('rec').onclick = () => command('record_toggle');
document.getElementById('camview').onclick = () => setCamView(!camView);
document.getElementById('handview').onclick = () => setHandView(handView === 'skin' ? 'robot' : 'skin');
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { command('record_toggle'); e.preventDefault(); }
  if (e.key === 'r') command('reset');
  if (e.key === 'c') showContacts = !showContacts;
  if (e.key === 'v') setCamView(!camView);
  if (e.key === 'h') setHandView(handView === 'skin' ? 'robot' : 'skin');
  if (e.key === 'n') command('next_scene');
  if (e.key === 'p') command('prev_scene');
});
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --------------------------------------------------------------------------
// WebXR
// --------------------------------------------------------------------------

let xrSession = null;
let xrMode = null;
let needsRecenter = false;
let tableOffset = 0;
let lastHandsSeen = 0;

async function initXRButtons() {
  const onQuest = /OculusBrowser|Quest/i.test(navigator.userAgent);
  if (!window.isSecureContext) {
    // Never silently disable the buttons: explain what to open instead.
    if (onQuest || !isLocalHost()) banner(secureContextHelp(), 'info');
  } else if (!navigator.xr && onQuest) {
    banner('This page has no WebXR access. Use the built-in Meta Quest Browser.');
  }
  for (const [id, mode] of [['enter-vr', 'immersive-vr'], ['enter-ar', 'immersive-ar']]) {
    const button = document.getElementById(id);
    button.onclick = () => startXR(mode);
    if (navigator.xr) {
      navigator.xr.isSessionSupported(mode).then((ok) => {
        button.title = ok ? '' : `${mode} is not supported by this browser/device`;
        if (!ok && mode === 'immersive-ar' && onQuest) button.style.display = 'none';
      }).catch(() => {});
    }
  }
}
initXRButtons();

async function requestSession(mode) {
  const features = { optionalFeatures: ['hand-tracking', 'bounded-floor'] };
  try {
    return { session: await navigator.xr.requestSession(mode, { requiredFeatures: ['local-floor'], ...features }), space: 'local-floor' };
  } catch (first) {
    // Some runtimes lack local-floor: fall back to local (seated) space.
    const session = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor', 'hand-tracking'] });
    return { session, space: 'local' };
  }
}

async function startXR(mode) {
  if (xrSession) { await xrSession.end(); return; }
  if (!window.isSecureContext) { banner(secureContextHelp()); return; }
  if (!navigator.xr) {
    banner('This browser has no WebXR support. On the headset use the Meta Quest Browser; on a PC, ' +
      'use the desktop view (the VR buttons need a headset).');
    return;
  }
  let result;
  try {
    result = await requestSession(mode);
  } catch (err) {
    banner(`Could not start ${mode === 'immersive-ar' ? 'passthrough' : 'VR'}: ${err.message || err}. ` +
      (isLocalHost() || /Quest/i.test(navigator.userAgent) ? '' : 'Is a headset connected?'));
    return;
  }
  hideBanner();
  xrSession = result.session;
  xrMode = mode;
  renderer.xr.setReferenceSpaceType(result.space);
  scene.background = mode === 'immersive-ar' ? null : background;
  kitchen.group.visible = mode !== 'immersive-ar';  // passthrough shows the real room
  renderer.xr.setFoveation(1.0);  // cheaper periphery on Quest
  await renderer.xr.setSession(xrSession);
  needsRecenter = true;
  lastHandsSeen = performance.now();
  send({ type: 'hello', role: 'headset' });
  xrSession.addEventListener('end', () => {
    xrSession = null;
    xrMode = null;
    scene.background = background;
    kitchen.group.visible = true;
    ghost.count = 0;
    send({ type: 'hands', left: { tracked: false }, right: { tracked: false } });
  });
}

// Place the physics origin on a table in front of the operator, yaw-aligned
// with the head (physics -Z points away from the operator).
function recenter(viewerPose) {
  const p = viewerPose.transform.position;
  const q = viewerPose.transform.orientation;
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(new THREE.Quaternion(q.x, q.y, q.z, q.w));
  forward.y = 0;
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1);
  forward.normalize();
  const yaw = Math.atan2(-forward.x, -forward.z);
  const m = new THREE.Matrix4().makeRotationY(yaw);
  m.setPosition(p.x + forward.x * TABLE_AHEAD, p.y - TABLE_BELOW_HEAD + tableOffset, p.z + forward.z * TABLE_AHEAD);
  setAnchor(m);
  panel.visible = true;
}

// In-headset button panel, poked with either index fingertip.
const panel = new THREE.Group();
panel.visible = false;
workspace.add(panel);
panel.position.set(-0.4, 0.2, 0.05);
panel.rotation.y = 0.45;
const buttons = [];
function makeButton(label, x, y, action, w = 0.085, h = 0.04) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 120;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false }));
  mesh.position.set(x, y, 0);
  panel.add(mesh);
  const b = { mesh, canvas, tex, label, action, w, h, cooldown: 0, lit: 0 };
  drawButton(b);
  buttons.push(b);
  return b;
}
function drawButton(b, text = b.label, active = false) {
  const ctx = b.canvas.getContext('2d');
  ctx.clearRect(0, 0, 256, 120);
  ctx.fillStyle = active ? '#c0392b' : (b.lit > 0 ? '#3d6fd8' : 'rgba(35,39,48,0.92)');
  ctx.beginPath(); ctx.roundRect(4, 4, 248, 112, 22); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = `bold ${text.length > 9 ? 34 : 44}px system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 62);
  b.tex.needsUpdate = true;
}
const recButton = makeButton('● Rec', -0.047, 0.05, () => command('record_toggle'));
makeButton('Reset', 0.047, 0.05, () => command('reset'));
makeButton('◀ Scene', -0.047, 0.0, () => command('prev_scene'));
makeButton('Scene ▶', 0.047, 0.0, () => command('next_scene'));
makeButton('Cam view', -0.047, -0.05, () => { tableOffset = 0; needsRecenter = true; });
makeButton('Contacts', 0.047, -0.05, () => { showContacts = !showContacts; });
makeButton('Table ▲', -0.047, -0.10, () => { tableOffset += 0.03; anchor.elements[13] += 0.03; setAnchor(anchor); });
makeButton('Table ▼', 0.047, -0.10, () => { tableOffset -= 0.03; anchor.elements[13] -= 0.03; setAnchor(anchor); });
makeButton('Ghost', -0.047, -0.15, () => { showGhost = !showGhost; });
makeButton('Hands', -0.047, -0.20, () => setHandView(handView === 'skin' ? 'robot' : 'skin'));
makeButton('Exit VR', 0.047, -0.15, () => { if (xrSession) xrSession.end(); });
const infoCanvas = document.createElement('canvas');
infoCanvas.width = 512; infoCanvas.height = 200;
const infoTex = new THREE.CanvasTexture(infoCanvas);
infoTex.colorSpace = THREE.SRGBColorSpace;
const info = new THREE.Mesh(new THREE.PlaneGeometry(0.18, 0.07),
  new THREE.MeshBasicMaterial({ map: infoTex, transparent: true, toneMapped: false }));
info.position.set(0, 0.118, 0);
panel.add(info);
let infoKey = '';
function drawInfo(lines) {
  const key = lines.join('|');
  if (key === infoKey) return;
  infoKey = key;
  const ctx = infoCanvas.getContext('2d');
  ctx.clearRect(0, 0, 512, 200);
  ctx.fillStyle = 'rgba(20,22,28,0.85)';
  ctx.beginPath(); ctx.roundRect(0, 0, 512, 200, 18); ctx.fill();
  lines.forEach((l, i) => {
    ctx.fillStyle = l.startsWith('!') ? '#ffb4a8' : '#e8e8ea';
    ctx.font = '28px system-ui, sans-serif';
    ctx.fillText(l.replace(/^!/, ''), 18, 42 + i * 46);
  });
  infoTex.needsUpdate = true;
}

const _local = new THREE.Vector3();
function pokeButtons(tips, dt) {
  for (const b of buttons) {
    b.cooldown = Math.max(0, b.cooldown - dt);
    if (b.lit > 0) { b.lit -= dt; if (b.lit <= 0) drawButton(b, b.label, b === recButton && statusInfo.recording); }
    for (const tip of tips) {
      _local.copy(tip);
      b.mesh.worldToLocal(_local);
      if (Math.abs(_local.x) < b.w / 2 && Math.abs(_local.y) < b.h / 2 && Math.abs(_local.z) < 0.012 && b.cooldown === 0) {
        b.cooldown = 0.8; b.lit = 0.25;
        drawButton(b);
        b.action();
      }
    }
  }
}

// --------------------------------------------------------------------------
// Hand streaming
// --------------------------------------------------------------------------

const poseBuffer = new Float32Array(16 * JOINTS);
const radiiBuffer = new Float32Array(JOINTS);
const _mat = new THREE.Matrix4();
const _inv = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scl = new THREE.Vector3();
let lastSend = 0;

function readHand(frame, refSpace, source, ghostOffset, tips) {
  const hand = source.hand;
  if (!hand || !frame.fillPoses) return { tracked: false };
  const joints = Array.from(hand.values());
  if (joints.length !== JOINTS || !frame.fillPoses(joints, refSpace, poseBuffer)) return { tracked: false };
  frame.fillJointRadii(joints, radiiBuffer);
  const p = new Array(3 * JOINTS), q = new Array(4 * JOINTS);
  for (let j = 0; j < JOINTS; j++) {
    _mat.fromArray(poseBuffer, 16 * j);
    _mat.decompose(_pos, _quat, _scl);
    ghost.setMatrixAt(ghostOffset + j, _mat);
    if (j === TIP_INDEX) tips.push(_pos.clone());
    _mat.premultiply(_inv);  // room -> physics frame
    _mat.decompose(_pos, _quat, _scl);
    p[3 * j] = _pos.x; p[3 * j + 1] = _pos.y; p[3 * j + 2] = _pos.z;
    q[4 * j] = _quat.x; q[4 * j + 1] = _quat.y; q[4 * j + 2] = _quat.z; q[4 * j + 3] = _quat.w;
  }
  return { tracked: true, p, q, r: Array.from(radiiBuffer) };
}

let lastTime = performance.now();
let recDrawn = null;
let fps = 0, fpsFrames = 0, fpsStart = performance.now();
renderer.setAnimationLoop((time, frame) => {
  const now = performance.now();
  const dt = (now - lastTime) / 1000;
  lastTime = now;
  fpsFrames++;
  if (now - fpsStart > 1000) { fps = fpsFrames * 1000 / (now - fpsStart); fpsFrames = 0; fpsStart = now; }
  let handHint = '';
  if (frame && xrSession) {
    const refSpace = renderer.xr.getReferenceSpace();
    const viewer = frame.getViewerPose(refSpace);
    if (viewer && needsRecenter) { recenter(viewer); needsRecenter = false; }
    _inv.copy(anchor).invert();
    const message = { type: 'hands', left: { tracked: false }, right: { tracked: false } };
    const tips = [];
    let offset = 0;
    let controllers = false;
    for (const source of xrSession.inputSources) {
      if (!source.hand) { if (source.gamepad) controllers = true; continue; }
      if (source.handedness !== 'left' && source.handedness !== 'right') continue;
      message[source.handedness] = readHand(frame, refSpace, source, offset, tips);
      if (message[source.handedness].tracked) offset += JOINTS;
    }
    if (offset > 0) lastHandsSeen = now;
    else if (now - lastHandsSeen > 2000) {
      handHint = controllers ? '!Put the controllers down: hands drive the sim'
        : '!No hands: enable Settings > Hand tracking';
    }
    ghost.count = showGhost ? offset : 0;
    ghost.instanceMatrix.needsUpdate = true;
    if (viewer) {
      _mat.fromArray(viewer.transform.matrix).premultiply(_inv);
      _mat.decompose(_pos, _quat, _scl);
      message.head = [_pos.x, _pos.y, _pos.z, _quat.x, _quat.y, _quat.z, _quat.w];
    }
    if (now - lastSend >= SEND_PERIOD_MS) { send(message); lastSend = now; }
    pokeButtons(tips, dt);
  } else if (camView) {
    updateCamView();
  } else {
    controls.update();
  }
  const h = lastHeader;
  const lines = [
    currentScene ? currentScene.name : '…',
    h ? `t ${h.sim_time.toFixed(1)}s  RTF ${h.rtf.toFixed(2)}  contacts ${h.total_contacts}  ${fps.toFixed(0)} fps` : '',
    statusInfo.recording ? `● REC ${h ? h.recorded_steps : 0} steps` : (connected ? 'idle' : '!not connected to the PC'),
  ];
  if (handHint) lines.push(handHint);
  drawInfo(lines);
  if (recDrawn !== statusInfo.recording) {
    recDrawn = statusInfo.recording;
    drawButton(recButton, statusInfo.recording ? '■ Stop' : '● Rec', statusInfo.recording);
  }
  const tracked = h ? Object.entries(h.tracked).map(([s, t]) => `${s}:${t ? 'tracked' : '—'}`).join(' ') : '';
  document.getElementById('status').textContent =
    `${lines.slice(0, 3).join('   ')}   hands ${tracked}${h && h.head ? '   headset connected' : ''}\n` +
    (now < flashUntil ? flashText : 'Space: record  R: reset  V: cam view  H: hand view  N/P: next/prev scene  C: contacts');
  renderer.render(scene, camera);
});

window.superdexAppReady = true;
