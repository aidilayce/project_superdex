// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// 3D environments around the teleoperation table (display only; physics
// only knows the table plane at y = 0). Built in the physics frame: the
// table / countertop top surface is y = 0, the operator stands at +Z, and
// the floor is `height` below (setHeight).
//
// Materials come from the CC0 Poly Haven asset pack when the PC has fetched
// it (albedo + OpenGL normal + roughness maps, real-world scale via
// world-space UVs); otherwise procedural stand-ins are used. Props are the
// pack's glTF models (real scale).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const ENVIRONMENTS = ['kitchen_sink', 'kitchen_island', 'studio'];

// --------------------------------------------------------------------------
// Materials
// --------------------------------------------------------------------------

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function canvasTexture(size, draw, srgb = true) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  draw(canvas.getContext('2d'), size);
  const t = new THREE.CanvasTexture(canvas);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  return t;
}

// Procedural fallbacks (used until / unless the Poly Haven pack exists).
const FALLBACK = {
  countertop: () => canvasTexture(1024, (ctx, s) => {
    const r = rng(11);
    ctx.fillStyle = '#e9e6e1';
    ctx.fillRect(0, 0, s, s);
    for (let i = 0; i < 9000; i++) {  // granite speckle
      const g = 150 + r() * 90;
      ctx.fillStyle = `rgba(${g},${g - 4},${g - 10},${0.15 + r() * 0.25})`;
      ctx.fillRect(r() * s, r() * s, 1 + r() * 2.5, 1 + r() * 2.5);
    }
    ctx.filter = 'blur(1.5px)';
    for (let i = 0; i < 10; i++) {  // soft veins
      ctx.strokeStyle = `rgba(150,148,150,${0.04 + r() * 0.07})`;
      ctx.lineWidth = 1 + r() * 2.5;
      ctx.beginPath();
      let x = r() * s, y = r() * s;
      ctx.moveTo(x, y);
      for (let k = 0; k < 5; k++) {
        const nx = x + (r() - 0.3) * 220, ny = y + (r() - 0.5) * 140;
        ctx.quadraticCurveTo((x + nx) / 2 + (r() - 0.5) * 80, (y + ny) / 2 + (r() - 0.5) * 80, nx, ny);
        x = nx; y = ny;
      }
      ctx.stroke();
    }
    ctx.filter = 'none';
  }),
  cabinet: () => canvasTexture(1024, (ctx, s) => {
    const r = rng(3);
    ctx.fillStyle = '#b08457';
    ctx.fillRect(0, 0, s, s);
    for (let i = 0; i < 400; i++) {
      ctx.strokeStyle = `rgba(${90 + r() * 40},${55 + r() * 25},${25 + r() * 15},${0.08 + r() * 0.12})`;
      ctx.lineWidth = 0.8 + r() * 2;
      const y = r() * s;
      ctx.beginPath(); ctx.moveTo(0, y);
      ctx.bezierCurveTo(s * 0.3, y + (r() - 0.5) * 14, s * 0.7, y + (r() - 0.5) * 14, s, y);
      ctx.stroke();
    }
  }),
  backsplash: () => canvasTexture(512, (ctx, s) => {
    const r = rng(5);
    ctx.fillStyle = '#cfd1d2';
    ctx.fillRect(0, 0, s, s);
    const tw = s / 2, th = s / 8;
    for (let row = 0; row < 8; row++) {
      for (let col = -1; col < 3; col++) {
        const x = col * tw + (row % 2) * tw / 2, y = row * th;
        const t = 238 + r() * 12;
        const grad = ctx.createLinearGradient(x, y, x, y + th);
        grad.addColorStop(0, `rgb(${t},${t},${t})`);
        grad.addColorStop(1, `rgb(${t - 12},${t - 11},${t - 10})`);
        ctx.fillStyle = grad;
        ctx.fillRect(x + 3, y + 3, tw - 6, th - 6);
      }
    }
  }),
  floor: () => canvasTexture(1024, (ctx, s) => {
    const r = rng(7);
    const plank = s / 6;
    for (let row = 0; row < 6; row++) {
      let x = -r() * 400;
      while (x < s) {
        const len = 350 + r() * 400, tone = 150 + r() * 30;
        ctx.fillStyle = `rgb(${tone + 30},${tone},${tone - 38})`;
        ctx.fillRect(x, row * plank, len, plank);
        for (let g = 0; g < 10; g++) {
          ctx.strokeStyle = `rgba(80,50,20,${0.05 + r() * 0.07})`;
          const y = row * plank + r() * plank;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + len, y + (r() - 0.5) * 6); ctx.stroke();
        }
        ctx.strokeStyle = 'rgba(40,25,10,0.5)'; ctx.lineWidth = 2;
        ctx.strokeRect(x, row * plank, len, plank);
        x += len;
      }
    }
  }),
  wall: () => canvasTexture(256, (ctx, s) => {
    const r = rng(9);
    ctx.fillStyle = '#ebe7df';
    ctx.fillRect(0, 0, s, s);
    for (let i = 0; i < 2500; i++) {
      ctx.fillStyle = `rgba(${r() > 0.5 ? 255 : 0},${r() > 0.5 ? 255 : 0},${r() > 0.5 ? 255 : 0},0.02)`;
      ctx.fillRect(r() * s, r() * s, 2, 2);
    }
  }),
  tabletop: () => canvasTexture(1024, (ctx, s) => {
    const r = rng(13);
    const strip = s / 16;
    for (let i = 0; i < 16; i++) {
      const t = 165 + r() * 35;
      ctx.fillStyle = `rgb(${t + 35},${t + 2},${t - 45})`;
      ctx.fillRect(0, i * strip, s, strip);
      for (let g = 0; g < 8; g++) {
        ctx.strokeStyle = `rgba(90,55,20,${0.05 + r() * 0.08})`;
        const y = i * strip + r() * strip;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(s, y + (r() - 0.5) * 5); ctx.stroke();
      }
    }
  }),
};

// Real-world tile size [m] of each material's texture.
const TILE = { countertop: 0.9, cabinet: 1.2, backsplash: 0.6, floor: 2.0, wall: 2.0, tabletop: 1.5, steel: 0.5 };
const PARAMS = {
  countertop: { roughness: 0.25, color: 0xffffff },
  cabinet: { roughness: 0.55 },
  backsplash: { roughness: 0.2 },
  floor: { roughness: 0.6 },
  wall: { roughness: 0.95 },
  tabletop: { roughness: 0.5 },
};

export class MaterialLibrary {
  constructor(pack) {
    this.pack = pack || null;
    this.loader = new THREE.TextureLoader();
    this.cache = new Map();
  }

  _load(url, srgb) {
    const t = this.loader.load(url);
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    return t;
  }

  get(slot) {
    if (this.cache.has(slot)) return this.cache.get(slot);
    let material;
    const packed = this.pack && this.pack.materials && this.pack.materials[slot];
    if (slot === 'steel' || slot === 'chrome') {
      material = new THREE.MeshStandardMaterial({
        color: slot === 'chrome' ? 0xf2f3f5 : 0xcfd2d6,
        metalness: 1.0,
        roughness: slot === 'chrome' ? 0.06 : 0.28,
      });
      if (slot === 'steel' && packed && packed.normal) {
        material.normalMap = this._load(packed.normal, false);
        material.normalScale = new THREE.Vector2(0.3, 0.3);
      }
    } else if (packed && packed.albedo) {
      material = new THREE.MeshStandardMaterial({
        map: this._load(packed.albedo, true),
        normalMap: packed.normal ? this._load(packed.normal, false) : null,
        roughnessMap: packed.roughness ? this._load(packed.roughness, false) : null,
        roughness: 1.0,
      });
    } else {
      material = new THREE.MeshStandardMaterial({ map: FALLBACK[slot](), ...(PARAMS[slot] || {}) });
    }
    material.userData.tile = TILE[slot] || 1.0;
    this.cache.set(slot, material);
    return material;
  }
}

// Box/plane UVs in meters (divided by the material's tile size), picked per
// face from its normal, so textures keep real-world scale on any size.
function worldUV(geometry, tile) {
  const p = geometry.getAttribute('position');
  const n = geometry.getAttribute('normal');
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    const ax = Math.abs(n.getX(i)), ay = Math.abs(n.getY(i)), az = Math.abs(n.getZ(i));
    let u, v;
    if (ay >= ax && ay >= az) { u = p.getX(i); v = p.getZ(i); }
    else if (ax >= az) { u = p.getZ(i); v = p.getY(i); }
    else { u = p.getX(i); v = p.getY(i); }
    uv[2 * i] = u / tile;
    uv[2 * i + 1] = v / tile;
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}

function box(parent, lib, slot, size, center, { shadow = true } = {}) {
  const material = typeof slot === 'string' ? lib.get(slot) : slot;
  const geometry = new THREE.BoxGeometry(...size);
  geometry.translate(...center);  // UVs in world meters
  worldUV(geometry, material.userData.tile || 1.0);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = shadow;
  mesh.castShadow = false;
  parent.add(mesh);
  return mesh;
}

function plane(parent, lib, slot, width, height, position, rotation) {
  const material = lib.get(slot);
  const geometry = new THREE.PlaneGeometry(width, height);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.rotation.set(...rotation);
  mesh.updateMatrix();
  geometry.applyMatrix4(mesh.matrix);
  mesh.position.set(0, 0, 0);
  mesh.rotation.set(0, 0, 0);
  worldUV(geometry, material.userData.tile || 1.0);
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

// --------------------------------------------------------------------------
// Props
// --------------------------------------------------------------------------

const gltfLoader = new GLTFLoader();
function addProps(parent, pack, spots) {
  const props = (pack && pack.props) || [];
  props.slice(0, spots.length).forEach((prop, i) => {
    gltfLoader.load(prop.url, (gltf) => {
      const model = gltf.scene;
      model.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      // Rest on the surface: lift by the model's lowest point.
      const bounds = new THREE.Box3().setFromObject(model);
      const [x, y, z, yaw] = spots[i];
      model.position.set(x, y - bounds.min.y, z);
      model.rotation.y = yaw;
      parent.add(model);
    });
  });
  return props.length;
}

function fallbackProps(parent, spots) {
  // A cutting board, a fruit bowl and a dish-soap bottle.
  const board = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.018, 0.22),
    new THREE.MeshStandardMaterial({ map: FALLBACK.tabletop(), roughness: 0.55 }));
  board.position.set(spots[0][0], spots[0][1] + 0.009, spots[0][2]);
  board.rotation.y = 0.2;
  const bowl = new THREE.Mesh(new THREE.SphereGeometry(0.11, 32, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.25, side: THREE.DoubleSide }));
  bowl.position.set(spots[1][0], spots[1][1] + 0.11, spots[1][2]);
  const group = new THREE.Group();
  group.add(board, bowl);
  [0xd23a2a, 0xf2b01e, 0x7cb342, 0xe0662a].forEach((color, i) => {
    const fruit = new THREE.Mesh(new THREE.SphereGeometry(0.035, 20, 14),
      new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
    fruit.position.set(spots[1][0] + Math.cos(i * 1.6) * 0.045, spots[1][1] + 0.04, spots[1][2] + Math.sin(i * 1.6) * 0.045);
    group.add(fruit);
  });
  for (const o of group.children) { o.castShadow = true; o.receiveShadow = true; }
  parent.add(group);
}

// --------------------------------------------------------------------------
// Kitchen sink: a stone countertop with an undermount stainless sink
// ("sponge in front of the sink"). The countertop is the physics table.
// --------------------------------------------------------------------------

// Defaults of workspace.py's kitchen_sink_layout (the server sends the
// authoritative numbers with every scene; its colliders use the same ones).
const SINK_DEFAULTS = {
  front: 0.3, back: -0.62, left: -1.4, right: 1.4, slab: 0.035,
  sink: [0.32, 0.82, -0.46, -0.06], sink_depth: 0.2,
};

function buildKitchenSink(lib, pack, params = {}) {
  const g = new THREE.Group();
  const P = { ...SINK_DEFAULTS, ...params };
  const { front, back, left, right, slab } = P;
  // Sink opening, to the operator's right within reach.
  const [sx0, sx1, sz0, sz1] = P.sink;
  const depth = P.sink_depth;
  const counter = lib.get('countertop');
  box(g, lib, counter, [sx0 - left, slab, front - back], [(left + sx0) / 2, -slab / 2, (front + back) / 2]);
  box(g, lib, counter, [right - sx1, slab, front - back], [(sx1 + right) / 2, -slab / 2, (front + back) / 2]);
  box(g, lib, counter, [sx1 - sx0, slab, front - sz1], [(sx0 + sx1) / 2, -slab / 2, (sz1 + front) / 2]);
  box(g, lib, counter, [sx1 - sx0, slab, sz0 - back], [(sx0 + sx1) / 2, -slab / 2, (back + sz0) / 2]);
  // Rounded front edge (bullnose).
  const nose = new THREE.Mesh(new THREE.CylinderGeometry(slab / 2, slab / 2, right - left, 24), counter);
  nose.rotation.z = Math.PI / 2;
  nose.position.set(0, -slab / 2, front);
  g.add(nose);

  // Stainless basin with a drain.
  const steel = lib.get('steel');
  const w = sx1 - sx0, d = sz1 - sz0, t = 0.008, cx = (sx0 + sx1) / 2, cz = (sz0 + sz1) / 2;
  box(g, lib, steel, [w, t, d], [cx, -depth, cz]);
  box(g, lib, steel, [t, depth, d], [sx0 + t / 2, -depth / 2 - slab / 2, cz]);
  box(g, lib, steel, [t, depth, d], [sx1 - t / 2, -depth / 2 - slab / 2, cz]);
  box(g, lib, steel, [w, depth, t], [cx, -depth / 2 - slab / 2, sz0 + t / 2]);
  box(g, lib, steel, [w, depth, t], [cx, -depth / 2 - slab / 2, sz1 - t / 2]);
  const drain = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.004, 32),
    new THREE.MeshStandardMaterial({ color: 0x3a3d42, metalness: 1, roughness: 0.35 }));
  drain.position.set(cx + 0.08, -depth + t, cz + 0.05);
  g.add(drain);

  // Gooseneck faucet (chrome) behind the basin.
  const chrome = lib.get('chrome');
  const fx = cx, fz = sz0 - 0.07;
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.032, 0.05, 32), chrome);
  base.position.set(fx, 0.025, fz);
  g.add(base);
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(fx, 0.04, fz), new THREE.Vector3(fx, 0.22, fz), new THREE.Vector3(fx, 0.33, fz + 0.04),
    new THREE.Vector3(fx, 0.34, fz + 0.12), new THREE.Vector3(fx, 0.27, fz + 0.2), new THREE.Vector3(fx, 0.22, fz + 0.21),
  ]);
  g.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 64, 0.014, 20, false), chrome));
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.016, 0.05, 24), chrome);
  head.position.set(fx, 0.2, fz + 0.21);
  g.add(head);
  const lever = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.09, 16), chrome);
  lever.position.set(fx + 0.045, 0.1, fz + 0.01);
  lever.rotation.z = Math.PI / 2.6;
  g.add(lever);
  for (const o of [base, head, lever]) o.castShadow = true;

  // Dish-soap bottle and a sponge holder next to the sink.
  const bottle = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.033, 0.18, 32),
    new THREE.MeshPhysicalMaterial({ color: 0x7fc9a0, roughness: 0.15, transmission: 0.0, clearcoat: 0.6 }));
  bottle.position.set(sx1 + 0.12, 0.09, sz0 + 0.02);
  const pump = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.012, 0.05, 16), chrome);
  pump.position.set(sx1 + 0.12, 0.2, sz0 + 0.02);
  g.add(bottle, pump);
  bottle.castShadow = true;

  // Backsplash tiles, plaster above, window over the sink.
  plane(g, lib, 'backsplash', right - left, 0.55, [0, 0.275, back + 0.001], [0, 0, 0]);
  plane(g, lib, 'wall', right - left + 4, 3, [0, 0.55 + 1.5, back], [0, 0, 0]);
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(0.72, 0.62),
    new THREE.MeshBasicMaterial({ color: 0xdfeefa, toneMapped: false }));
  glass.position.set(cx, 0.98, back + 0.004);
  g.add(glass);
  const frame = lib.get('wall');
  for (const [bw, bh, bx, by] of [[0.8, 0.05, cx, 1.31], [0.8, 0.05, cx, 0.65], [0.05, 0.7, cx - 0.385, 0.98], [0.05, 0.7, cx + 0.385, 0.98], [0.03, 0.62, cx, 0.98]]) {
    box(g, lib, frame, [bw, bh, 0.06], [bx, by, back + 0.03]);
  }
  const sill = box(g, lib, counter, [0.86, 0.02, 0.12], [cx, 0.64, back + 0.06]);
  sill.castShadow = true;

  // Upper cabinets left of the window.
  const cab = lib.get('cabinet');
  box(g, lib, cab, [cx - 0.45 - left, 0.72, 0.34], [(left + cx - 0.45) / 2, 1.1, back + 0.17]);
  const handleMat = lib.get('steel');
  for (let x = left + 0.3; x < cx - 0.5; x += 0.45) {
    const h = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.14, 12), handleMat);
    h.position.set(x, 0.82, back + 0.35);
    g.add(h);
  }

  // Base cabinets (height follows the counter height, see setHeight).
  const baseCab = new THREE.Group();
  g.add(baseCab);
  const kitchenFloor = new THREE.Group();
  g.add(kitchenFloor);
  plane(kitchenFloor, lib, 'floor', 8, 8, [0, 0, 1.5], [-Math.PI / 2, 0, 0]);
  plane(kitchenFloor, lib, 'wall', 8, 3.2, [left - 0.02, 1.6, 1.5], [0, Math.PI / 2, 0]);
  plane(kitchenFloor, lib, 'wall', 8, 3.2, [3.2, 1.6, 1.5], [0, -Math.PI / 2, 0]);
  plane(kitchenFloor, lib, 'wall', 8, 3.2, [0, 1.6, 4.0], [0, Math.PI, 0]);

  function setHeight(height) {
    baseCab.clear();
    const h = Math.max(0.3, height) - slab;
    // Carcass below the basin (the sink opening must show the steel basin),
    // plus the door fronts up to the countertop.
    const under = depth + 0.03 - slab;
    box(baseCab, lib, cab, [right - left, h - 0.1 - under, front - back - 0.06],
      [0, -slab - under - (h - 0.1 - under) / 2, (front + back) / 2 - 0.03]);
    box(baseCab, lib, cab, [right - left, under, 0.02], [0, -slab - under / 2, front - 0.07]);
    box(baseCab, lib, lib.get('floor'), [right - left, 0.1, front - back - 0.12], [0, -height + 0.05, (front + back) / 2 - 0.06]);
    // Door gaps and handles on the fronts.
    const gap = new THREE.MeshStandardMaterial({ color: 0x2a1f16, roughness: 0.8 });
    for (let x = left + 0.6; x < right - 0.1; x += 0.6) {
      box(baseCab, lib, gap, [0.006, h - 0.14, 0.004], [x, -slab - (h - 0.1) / 2, front - 0.058]);
    }
    for (let x = left + 0.3; x < right; x += 0.6) {
      const hdl = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.16, 12), handleMat);
      hdl.rotation.z = Math.PI / 2;
      hdl.position.set(x, -slab - 0.08, front - 0.045);
      baseCab.add(hdl);
    }
    kitchenFloor.position.y = -height;
  }

  // Decorative props (display only: kept out of reach, since they don't collide).
  const spots = [[-1.05, 0, -0.42, 0.3], [-0.78, 0, -0.5, 1.2], [1.2, 0, -0.35, 2.0], [-1.25, 0, -0.1, 0.8],
    [1.2, 0, 0.1, 0.4], [-1.3, 0, -0.5, 2.6]];
  if (!addProps(g, pack, spots)) fallbackProps(g, spots);
  return { group: g, setHeight, lamp: [cx - 0.6, 1.4, -0.1] };
}

// --------------------------------------------------------------------------
// Studio: just the table in SuperDex Studio's HDR.
// --------------------------------------------------------------------------

function buildStudio(lib, params = {}) {
  const g = new THREE.Group();
  const width = params.width || 1.4, depthZ = params.depth || 0.9;
  const top = box(g, lib, 'tabletop', [width, 0.04, depthZ], [0, -0.02, 0]);
  top.receiveShadow = true;
  const legs = new THREE.Group();
  g.add(legs);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x222428, metalness: 0.6, roughness: 0.4 });
  function setHeight(height) {
    legs.clear();
    for (const [x, z] of [[-0.64, -0.39], [0.64, -0.39], [-0.64, 0.39], [0.64, 0.39]]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.04, height - 0.04, 0.04), legMat);
      leg.position.set(x, -0.04 - (height - 0.04) / 2, z);
      legs.add(leg);
    }
  }
  return { group: g, setHeight, background: 'studio' };
}

// An imported room (rooms.py): the model, placed with the same transform as
// its physics colliders. It keeps its modeled height (fixedHeight).
function buildRoom(layout) {
  const group = new THREE.Group();
  const holder = new THREE.Group();
  holder.matrixAutoUpdate = false;
  holder.matrix.fromArray(layout.physics_from_model);
  group.add(holder);
  new GLTFLoader().load(layout.model_url, (gltf) => {
    gltf.scene.traverse((o) => {
      if (o.isMesh) { o.receiveShadow = true; o.castShadow = false; o.frustumCulled = true; }
    });
    holder.add(gltf.scene);
  }, undefined, (err) => console.error('room model', err));
  return { group, setHeight() {}, fixedHeight: layout.counter_height, attribution: layout.attribution };
}

export function buildEnvironment(name, lib, pack, params = {}, layout = null) {
  if (layout && layout.kind === 'room' && layout.name === name) return buildRoom(layout);
  if (name === 'studio') return buildStudio(lib, params);
  return buildKitchenSink(lib, pack, params);
}

// Wireframes of the server's static colliders (debug view, key C): what the
// simulated hands and objects actually touch.
export function buildColliderView(layout) {
  const group = new THREE.Group();
  if (!layout) return group;
  const material = new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.8, depthTest: false });
  if (layout.colliders_url) {
    // Imported room: its convex collider pieces.
    new GLTFLoader().load(layout.colliders_url, (gltf) => {
      gltf.scene.traverse((o) => {
        if (!o.isMesh) return;
        const lines = new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 20), material);
        o.updateWorldMatrix(true, false);
        lines.applyMatrix4(o.matrixWorld);
        lines.renderOrder = 10;
        group.add(lines);
      });
    });
  }
  for (const b of layout.boxes || []) {
    // Colliders reaching below the floor are drawn down to it.
    const bottom = Math.max(b.center[1] - b.size[1] / 2, -layout.counter_height);
    const top = b.center[1] + b.size[1] / 2;
    if (top <= bottom) continue;
    const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(b.size[0], top - bottom, b.size[2]));
    const lines = new THREE.LineSegments(geometry, material);
    lines.position.set(b.center[0], (top + bottom) / 2, b.center[2]);
    lines.renderOrder = 10;
    group.add(lines);
  }
  const floor = new THREE.GridHelper(4, 16, 0x00e5ff, 0x00e5ff);
  floor.position.y = -layout.counter_height;
  floor.material.transparent = true;
  floor.material.opacity = 0.5;
  group.add(floor);
  return group;
}

// --------------------------------------------------------------------------
// Sponge material: yellow foam with a green scrub layer; the pores stick to
// the (deforming) sponge through its rest positions.
// --------------------------------------------------------------------------

export function spongeMaterial(geometry) {
  const rest = geometry.getAttribute('position').clone();
  geometry.setAttribute('restPosition', rest);
  geometry.computeBoundingBox();
  const top = geometry.boundingBox.max.y;
  const pores = canvasTexture(256, (ctx, s) => {
    const r = rng(21);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, s, s);
    for (let i = 0; i < 1400; i++) {
      ctx.fillStyle = `rgba(120,90,20,${0.25 + r() * 0.35})`;
      ctx.beginPath();
      ctx.arc(r() * s, r() * s, 0.8 + r() * 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  const material = new THREE.MeshStandardMaterial({ color: 0xf5c83a, roughness: 0.95 });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.poreMap = { value: pores };
    shader.uniforms.scrubTop = { value: top };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec3 restPosition;\nvarying vec3 vRest;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvRest = restPosition;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D poreMap;\nuniform float scrubTop;\nvarying vec3 vRest;')
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 p = vRest * 18.0;
          float pore = (texture2D(poreMap, p.xy).r + texture2D(poreMap, p.yz).r + texture2D(poreMap, p.xz).r) / 3.0;
          diffuseColor.rgb *= mix(0.55, 1.0, pore);
          if (vRest.y > scrubTop - 0.011) diffuseColor.rgb = vec3(0.13, 0.42, 0.2) * mix(0.6, 1.0, pore);
        }`);
  };
  material.customProgramCacheKey = () => 'superdex-sponge';
  return material;
}
