// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// SuperDex Quest teleop client. In the Quest browser it runs a WebXR session
// with hand tracking, streams both 25-joint hand skeletons (in the physics
// frame) to the PC, and renders the simulated scene the PC streams back. On
// a desktop browser the same page is a spectator view with the same
// controls.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const JOINTS = 25;
const TIP_INDEX = 9;              // index-finger-tip
const SEND_PERIOD_MS = 1000 / 72; // hand message rate cap
const TABLE_BELOW_HEAD = 0.5;     // [m] default table height below the eyes
const TABLE_AHEAD = 0.42;         // [m] table center in front of the eyes

// --------------------------------------------------------------------------
// Renderer and scene graph
// --------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const background = new THREE.Color(0x15171c);
scene.background = background;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.6, 2.0, 1.0);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 50);
camera.position.set(0, 1.35, 0.75);
const controls = new OrbitControls(camera, renderer.domElement);

// Physics frame (Y-up, meters, table top at y = 0). Its placement in the
// room is the workspace anchor; everything simulated lives under it.
const workspace = new THREE.Group();
workspace.matrixAutoUpdate = false;
scene.add(workspace);
let anchor = new THREE.Matrix4().makeTranslation(0, 0.8, 0);
setAnchor(anchor);
controls.target.set(0, 0.8, 0);
controls.update();

function setAnchor(m) {
  anchor = m.clone();
  workspace.matrix.copy(anchor);
  workspace.matrixWorldNeedsUpdate = true;
}

const table = new THREE.Mesh(
  new THREE.BoxGeometry(1.0, 0.03, 0.7),
  new THREE.MeshStandardMaterial({ color: 0x6b5b4b, roughness: 0.85 }));
table.position.y = -0.015;
workspace.add(table);

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
  new THREE.MeshBasicMaterial({ color: 0x7fd3ff, transparent: true, opacity: 0.55 }), 2 * JOINTS);
ghost.count = 0;
ghost.frustumCulled = false;
scene.add(ghost);

// --------------------------------------------------------------------------
// Scene geometry from the server
// --------------------------------------------------------------------------

let actors = [];          // index -> {object, deformable}
let sceneList = [];
let currentScene = null;
let lastHeader = null;
let statusInfo = { recording: false };

const PALETTE = [0xe4572e, 0x29a19c, 0xf3a712, 0x6c8ead, 0xa8c686, 0xd183c9, 0x4f86c6, 0xf08a4b];
function colorFor(name) {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  if (/block_red|red/i.test(name)) return 0xd64541;
  if (/green/i.test(name)) return 0x4caf50;
  if (/blue/i.test(name)) return 0x3f7fd6;
  if (/yellow/i.test(name)) return 0xf2c230;
  return PALETTE[h % PALETTE.length];
}

function buildGeometry(msg) {
  actorRoot.clear();
  actors = [];
  currentScene = msg.scene;
  document.getElementById('desc').textContent = msg.scene.description || '';
  const sel = document.getElementById('scene');
  sel.value = msg.scene.id;
  for (const a of msg.actors) {
    const entry = { object: null, deformable: a.deformable };
    if (a.mesh === 'surface' || a.mesh === 'visual') {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(a.positions), 3));
      g.setIndex(a.indices);
      g.computeVertexNormals();
      const hand = a.hand !== '';
      const mat = new THREE.MeshStandardMaterial({
        color: hand ? 0xd9a88b : (a.static ? 0x9aa0a8 : colorFor(a.name)),
        roughness: hand ? 0.6 : 0.5,
        metalness: 0.0,
        transparent: hand,
        opacity: hand ? 0.92 : 1.0,
        side: a.deformable ? THREE.DoubleSide : THREE.FrontSide,
        flatShading: !a.deformable && !hand,
      });
      const mesh = new THREE.Mesh(g, mat);
      mesh.frustumCulled = false;
      if (a.deformable) mesh.matrixAutoUpdate = false;  // world-space vertices
      actorRoot.add(mesh);
      entry.object = mesh;
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
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => ws.send(JSON.stringify({ type: 'hello', role: xrSession ? 'headset' : 'viewer' }));
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
  ws.onclose = () => { flash('disconnected, retrying…'); setTimeout(connect, 1000); };
}
connect();

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}
function command(cmd, extra = {}) { send({ type: 'cmd', cmd, ...extra }); }

let flashText = '', flashUntil = 0;
function flash(text) { flashText = text; flashUntil = performance.now() + 4000; }

// --------------------------------------------------------------------------
// Desktop controls
// --------------------------------------------------------------------------

document.getElementById('scene').onchange = (e) => command('load_scene', { scene: e.target.value });
document.getElementById('reset').onclick = () => command('reset');
document.getElementById('rec').onclick = () => command('record_toggle');
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { command('record_toggle'); e.preventDefault(); }
  if (e.key === 'r') command('reset');
  if (e.key === 'c') showContacts = !showContacts;
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
let needsRecenter = false;
let tableOffset = 0;

async function setupXRButtons() {
  if (!navigator.xr) {
    flash(window.isSecureContext ? 'WebXR not available in this browser'
      : 'WebXR needs https:// or http://localhost (use adb reverse or --https)');
    return;
  }
  for (const [id, mode] of [['enter-vr', 'immersive-vr'], ['enter-ar', 'immersive-ar']]) {
    const button = document.getElementById(id);
    if (await navigator.xr.isSessionSupported(mode).catch(() => false)) {
      button.disabled = false;
      button.onclick = () => startXR(mode);
    }
  }
}
setupXRButtons();

async function startXR(mode) {
  if (xrSession) { xrSession.end(); return; }
  const session = await navigator.xr.requestSession(mode, {
    requiredFeatures: ['local-floor'],
    optionalFeatures: ['hand-tracking'],
  });
  xrSession = session;
  scene.background = mode === 'immersive-ar' ? null : background;
  await renderer.xr.setSession(session);
  needsRecenter = true;
  send({ type: 'hello', role: 'headset' });
  session.addEventListener('end', () => {
    xrSession = null;
    scene.background = background;
    ghost.count = 0;
    send({ type: 'hands', left: { tracked: false }, right: { tracked: false } });
  });
}

// Place the physics origin on a virtual table in front of the operator,
// yaw-aligned with the head (physics -Z points away from the operator).
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
panel.position.set(-0.38, 0.22, 0.05);
panel.rotation.y = 0.45;
const buttons = [];
function makeButton(label, x, y, action, w = 0.085, h = 0.04) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 120;
  const tex = new THREE.CanvasTexture(canvas);
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true }));
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
makeButton('Recenter', -0.047, -0.05, () => { tableOffset = 0; needsRecenter = true; });
makeButton('Contacts', 0.047, -0.05, () => { showContacts = !showContacts; });
makeButton('Table ▲', -0.047, -0.10, () => { tableOffset += 0.03; anchor.elements[13] += 0.03; setAnchor(anchor); });
makeButton('Table ▼', 0.047, -0.10, () => { tableOffset -= 0.03; anchor.elements[13] -= 0.03; setAnchor(anchor); });
const infoCanvas = document.createElement('canvas');
infoCanvas.width = 512; infoCanvas.height = 160;
const infoTex = new THREE.CanvasTexture(infoCanvas);
const info = new THREE.Mesh(new THREE.PlaneGeometry(0.18, 0.056), new THREE.MeshBasicMaterial({ map: infoTex, transparent: true }));
info.position.set(0, 0.11, 0);
panel.add(info);
let infoKey = '';
function drawInfo(lines) {
  const key = lines.join('|');
  if (key === infoKey) return;
  infoKey = key;
  const ctx = infoCanvas.getContext('2d');
  ctx.clearRect(0, 0, 512, 160);
  ctx.fillStyle = 'rgba(20,22,28,0.85)';
  ctx.beginPath(); ctx.roundRect(0, 0, 512, 160, 18); ctx.fill();
  ctx.fillStyle = '#e8e8ea';
  ctx.font = '30px system-ui, sans-serif';
  lines.forEach((l, i) => ctx.fillText(l, 18, 44 + i * 46));
  infoTex.needsUpdate = true;
}

const _tip = new THREE.Vector3();
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
  _inv.copy(anchor).invert();
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
renderer.setAnimationLoop((time, frame) => {
  const now = performance.now();
  const dt = (now - lastTime) / 1000;
  lastTime = now;
  if (frame && xrSession) {
    const refSpace = renderer.xr.getReferenceSpace();
    const viewer = frame.getViewerPose(refSpace);
    if (viewer && needsRecenter) { recenter(viewer); needsRecenter = false; }
    const message = { type: 'hands', left: { tracked: false }, right: { tracked: false } };
    const tips = [];
    let offset = 0;
    for (const source of xrSession.inputSources) {
      if (!source.hand || (source.handedness !== 'left' && source.handedness !== 'right')) continue;
      message[source.handedness] = readHand(frame, refSpace, source, offset, tips);
      if (message[source.handedness].tracked) offset += JOINTS;
    }
    ghost.count = offset;
    ghost.instanceMatrix.needsUpdate = true;
    if (viewer) {
      _mat.fromArray(viewer.transform.matrix).premultiply(_inv.copy(anchor).invert());
      _mat.decompose(_pos, _quat, _scl);
      message.head = [_pos.x, _pos.y, _pos.z, _quat.x, _quat.y, _quat.z, _quat.w];
    }
    if (now - lastSend >= SEND_PERIOD_MS) { send(message); lastSend = now; }
    pokeButtons(tips, dt);
  } else {
    controls.update();
  }
  const h = lastHeader;
  const lines = [
    currentScene ? currentScene.name : '…',
    h ? `t ${h.sim_time.toFixed(1)}s  RTF ${h.rtf.toFixed(2)}  contacts ${h.total_contacts}` : '',
    statusInfo.recording ? `● REC ${h ? h.recorded_steps : 0} steps` : 'idle',
  ];
  drawInfo(lines);
  if (recButton) drawButtonIfChanged();
  const tracked = h ? Object.entries(h.tracked).map(([s, t]) => `${s}:${t ? 'tracked' : '—'}`).join(' ') : '';
  document.getElementById('status').textContent =
    `${lines.join('   ')}   hands ${tracked}\n` +
    (now < flashUntil ? flashText : 'Space: record  R: reset  N/P: next/prev scene  C: contacts');
  renderer.render(scene, camera);
});

let recDrawn = null;
function drawButtonIfChanged() {
  if (recDrawn === statusInfo.recording) return;
  recDrawn = statusInfo.recording;
  drawButton(recButton, statusInfo.recording ? '■ Stop' : '● Rec', statusInfo.recording);
}
