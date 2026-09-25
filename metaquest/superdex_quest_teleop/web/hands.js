// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// Continuous skinned display hands. Each hand is a rigged mesh whose 25 bones
// carry the WebXR joint names (by default the WebXR generic-hand model, MIT);
// every frame its bones take the 25 joint poses the PC derives from the
// *simulated* Meta XR hand, so the skin shows exactly what the physics does.
//
// Models without their own textures get a procedural skin: UV-mapped albedo
// with pores, creases and blotches, a matching normal map, a subtle sheen and
// fingernails painted from the skinning weights. A textured model (e.g. a
// hand exported from a BEDLAM / SMPL-X avatar) keeps its own materials.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const JOINT_NAMES = [
  'wrist',
  'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  ...['index', 'middle', 'ring', 'pinky'].flatMap((f) => [
    `${f}-finger-metacarpal`, `${f}-finger-phalanx-proximal`, `${f}-finger-phalanx-intermediate`,
    `${f}-finger-phalanx-distal`, `${f}-finger-tip`]),
];

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// Tileable skin height field (pores, creases, micro noise) in [0, 1].
function skinHeight(size) {
  const r = rng(1234);
  const h = new Float32Array(size * size).fill(0.5);
  const splat = (cx, cy, radius, amount) => {
    const r0 = Math.ceil(radius);
    for (let dy = -r0; dy <= r0; dy++) {
      for (let dx = -r0; dx <= r0; dx++) {
        const d2 = (dx * dx + dy * dy) / (radius * radius);
        if (d2 > 1) continue;
        const x = (cx + dx + size) % size, y = (cy + dy + size) % size;
        h[y * size + x] += amount * (1 - d2) * (1 - d2);
      }
    }
  };
  for (let i = 0; i < 7000; i++) splat(Math.floor(r() * size), Math.floor(r() * size), 1 + r() * 1.6, -0.18 - r() * 0.2);
  for (let i = 0; i < 900; i++) {  // fine creases: short random walks
    let x = r() * size, y = r() * size;
    const a = r() * Math.PI;
    for (let k = 0; k < 14 + r() * 20; k++) {
      splat(Math.floor(x), Math.floor(y), 0.9, -0.05);
      x += Math.cos(a) + (r() - 0.5) * 0.6;
      y += Math.sin(a) + (r() - 0.5) * 0.6;
    }
  }
  for (let i = 0; i < h.length; i++) h[i] = Math.min(1, Math.max(0, h[i] + (r() - 0.5) * 0.06));
  return h;
}

function skinTextures() {
  const size = 512;
  const height = skinHeight(size);
  const r = rng(99);
  // Albedo: warm base with low-frequency blotches (capillaries, freckles).
  const albedo = document.createElement('canvas');
  albedo.width = albedo.height = size;
  const actx = albedo.getContext('2d');
  actx.fillStyle = 'rgb(255,255,255)';
  actx.fillRect(0, 0, size, size);
  for (let i = 0; i < 140; i++) {
    const x = r() * size, y = r() * size, rad = 18 + r() * 60;
    for (const [ox, oy] of [[0, 0], [size, 0], [-size, 0], [0, size], [0, -size]]) {
      const g = actx.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, rad);
      g.addColorStop(0, r() > 0.45 ? 'rgba(200,100,90,0.045)' : 'rgba(255,225,190,0.05)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      actx.fillStyle = g;
      actx.fillRect(x + ox - rad, y + oy - rad, 2 * rad, 2 * rad);
    }
  }
  const img = actx.getImageData(0, 0, size, size);
  for (let i = 0; i < size * size; i++) {  // pores darken the albedo slightly
    const shade = 0.93 + 0.07 * Math.min(1, height[i] * 2);
    img.data[4 * i] *= shade;
    img.data[4 * i + 1] *= shade * 0.985;
    img.data[4 * i + 2] *= shade * 0.97;
  }
  actx.putImageData(img, 0, 0);
  // Normal map from the height field (Sobel).
  const normal = document.createElement('canvas');
  normal.width = normal.height = size;
  const nctx = normal.getContext('2d');
  const nimg = nctx.createImageData(size, size);
  const at = (x, y) => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const dy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      const n = new THREE.Vector3(-dx * 1.4, -dy * 1.4, 1).normalize();
      const i = 4 * (y * size + x);
      nimg.data[i] = (n.x * 0.5 + 0.5) * 255;
      nimg.data[i + 1] = (n.y * 0.5 + 0.5) * 255;
      nimg.data[i + 2] = (n.z * 0.5 + 0.5) * 255;
      nimg.data[i + 3] = 255;
    }
  }
  nctx.putImageData(nimg, 0, 0);
  const map = new THREE.CanvasTexture(albedo);
  map.colorSpace = THREE.SRGBColorSpace;
  const normalMap = new THREE.CanvasTexture(normal);
  for (const t of [map, normalMap]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(6, 6);
    t.anisotropy = 4;
  }
  return { map, normalMap };
}

let textures = null;
function proceduralSkin(tone) {
  textures = textures || skinTextures();
  return new THREE.MeshPhysicalMaterial({
    color: tone,
    map: textures.map,
    normalMap: textures.normalMap,
    normalScale: new THREE.Vector2(0.25, 0.25),
    roughness: 0.5,
    metalness: 0.0,
    sheen: 0.12,
    sheenRoughness: 0.6,
    sheenColor: new THREE.Color(0xc07060),
    vertexColors: true,
  });
}

// Fingernails: vertices driven by a distal phalanx / tip bone, on its dorsal
// side and in its outer part, get a pale pink tint (as vertex colors).
function paintNails(mesh, tone) {
  const geometry = mesh.geometry;
  const skeleton = mesh.skeleton;
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  const skinIndex = geometry.getAttribute('skinIndex');
  const skinWeight = geometry.getAttribute('skinWeight');
  const bind = skeleton.boneInverses.map((m) => m.clone().invert());
  const byName = new Map(skeleton.bones.map((b, i) => [b.name, i]));
  const skinColor = new THREE.Color(tone);
  const nail = new THREE.Color(0xecc6bb);
  const tint = new THREE.Color(nail.r / skinColor.r, nail.g / skinColor.g, nail.b / skinColor.b);
  const colors = new Float32Array(position.count * 3).fill(1);
  const v = new THREE.Vector3(), n = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), y = new THREE.Vector3();
  for (let i = 0; i < position.count; i++) {
    let best = 0, bw = -1;
    for (let k = 0; k < 4; k++) {
      const w = skinWeight.getComponent(i, k);
      if (w > bw) { bw = w; best = skinIndex.getComponent(i, k); }
    }
    const name = skeleton.bones[best].name;
    if (!/(phalanx-distal|tip)$/.test(name)) continue;
    const distal = byName.get(name.replace(/tip$/, 'phalanx-distal'));
    const tip = byName.get(name.replace(/phalanx-distal$/, 'tip'));
    if (distal === undefined || tip === undefined) continue;
    a.setFromMatrixPosition(bind[distal]);
    b.setFromMatrixPosition(bind[tip]);
    const axis = b.clone().sub(a);
    const length = axis.length();
    axis.normalize();
    y.setFromMatrixColumn(bind[distal], 1).normalize();  // dorsal
    v.fromBufferAttribute(position, i).sub(a);
    n.fromBufferAttribute(normal, i);
    const t = v.dot(axis) / length;
    const up = n.dot(y);
    if (t < 0.25 || up < 0.35 || v.dot(y) < 0) continue;
    const w = Math.min(1, (t - 0.25) / 0.2) * Math.min(1, (up - 0.35) / 0.25);
    const edge = t > 1.0 ? 1.08 : 1.0;  // whiter free edge
    colors[3 * i] = 1 + (tint.r * edge - 1) * w;
    colors[3 * i + 1] = 1 + (tint.g * edge - 1) * w;
    colors[3 * i + 2] = 1 + (tint.b * edge - 1) * w;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

export class SkinnedHand {
  constructor(url, { tone = 0xc98f6e } = {}) {
    this.group = new THREE.Group();
    this.group.visible = false;
    this.bones = null;
    this.ready = new Promise((resolve, reject) => {
      new GLTFLoader().load(url, (gltf) => {
        const root = gltf.scene;
        let mesh = null;
        root.traverse((o) => { if (o.isSkinnedMesh && !mesh) mesh = o; });
        if (!mesh) { reject(new Error(`${url}: no skinned mesh`)); return; }
        const textured = [].concat(mesh.material).some((m) => m && m.map);
        if (!textured) {
          paintNails(mesh, tone);
          mesh.material = proceduralSkin(tone);
        }
        mesh.frustumCulled = false;
        this.bones = JOINT_NAMES.map((name) => root.getObjectByName(name) || null);
        this.group.add(root);
        // Forearm behind the wrist (WebXR wrist frame: -Z toward the fingers,
        // +Y dorsal, X across the wrist); display only. It fades out toward
        // the elbow instead of ending in a visible cut.
        const length = 0.26;
        const geometry = new THREE.CylinderGeometry(1, 1.25, length, 40, 8, true);
        geometry.rotateX(Math.PI / 2);                // axis along Z
        geometry.translate(0, 0, length / 2 - 0.012); // extend toward +Z (elbow)
        const fade = document.createElement('canvas');
        fade.width = 4; fade.height = 128;
        const fctx = fade.getContext('2d');
        const grad = fctx.createLinearGradient(0, 0, 0, 128);
        grad.addColorStop(0.0, '#000');  // elbow end (v = 1 at the top row)
        grad.addColorStop(0.55, '#fff');
        grad.addColorStop(1.0, '#fff');
        fctx.fillStyle = grad;
        fctx.fillRect(0, 0, 4, 128);
        const forearmMaterial = [].concat(mesh.material)[0].clone();
        forearmMaterial.vertexColors = false;
        forearmMaterial.alphaMap = new THREE.CanvasTexture(fade);
        forearmMaterial.transparent = true;
        forearmMaterial.side = THREE.DoubleSide;
        this.forearm = new THREE.Mesh(geometry, forearmMaterial);
        this.forearm.scale.set(0.029, 0.02, 1);
        this.forearm.frustumCulled = false;
        this.forearmHolder = new THREE.Group();
        this.forearmHolder.add(this.forearm);
        this.group.add(this.forearmHolder);
        resolve(this);
      }, undefined, reject);
    });
  }

  // joints: Float32Array view of 25 x [px py pz qx qy qz qw] in the parent frame.
  setJoints(joints) {
    if (!this.bones) return;
    for (let j = 0; j < 25; j++) {
      const bone = this.bones[j];
      if (!bone) continue;
      const k = 7 * j;
      bone.position.set(joints[k], joints[k + 1], joints[k + 2]);
      bone.quaternion.set(joints[k + 3], joints[k + 4], joints[k + 5], joints[k + 6]);
    }
    if (this.forearmHolder) {
      this.forearmHolder.position.set(joints[0], joints[1], joints[2]);
      this.forearmHolder.quaternion.set(joints[3], joints[4], joints[5], joints[6]);
    }
    this.group.visible = true;
  }
}
