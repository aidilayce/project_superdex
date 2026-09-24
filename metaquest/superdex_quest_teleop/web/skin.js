// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// Skin material for the simulated hands. The Meta XR hand render meshes carry
// no UVs, so a procedural skin texture (pores, fine creases, subtle color
// variation) is applied with triplanar mapping in the mesh's own frame: the
// texture sticks to each phalanx as it moves.

import * as THREE from 'three';

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function skinTexture() {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const r = rng(42);
  // Near-white base: the texture modulates the material color.
  ctx.fillStyle = 'rgb(246,240,236)';
  ctx.fillRect(0, 0, size, size);
  // Blotchy warmth (capillaries) with soft radial spots, wrapped for tiling.
  for (let i = 0; i < 90; i++) {
    const x = r() * size, y = r() * size, rad = 20 + r() * 70;
    for (const [ox, oy] of [[0, 0], [size, 0], [-size, 0], [0, size], [0, -size]]) {
      const g = ctx.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, rad);
      const warm = r() > 0.5;
      g.addColorStop(0, warm ? 'rgba(214,120,110,0.10)' : 'rgba(255,236,214,0.12)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(x + ox - rad, y + oy - rad, 2 * rad, 2 * rad);
    }
  }
  // Fine creases.
  for (let i = 0; i < 260; i++) {
    ctx.strokeStyle = `rgba(120,70,60,${0.05 + r() * 0.09})`;
    ctx.lineWidth = 0.6 + r() * 0.9;
    const x = r() * size, y = r() * size, a = r() * Math.PI, len = 8 + r() * 26;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.quadraticCurveTo(x + Math.cos(a) * len * 0.5 + (r() - 0.5) * 6, y + Math.sin(a) * len * 0.5 + (r() - 0.5) * 6,
      x + Math.cos(a) * len, y + Math.sin(a) * len);
    ctx.stroke();
  }
  // Pores.
  for (let i = 0; i < 5200; i++) {
    ctx.fillStyle = `rgba(110,60,50,${0.06 + r() * 0.12})`;
    const s = 0.6 + r() * 1.3;
    ctx.fillRect(r() * size, r() * size, s, s);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 4;
  return texture;
}

let sharedTexture = null;

// tone: base skin color; scale: texture repeats per meter.
export function makeSkinMaterial({ tone = 0xc98f6e, scale = 55, opacity = 1.0 } = {}) {
  sharedTexture = sharedTexture || skinTexture();
  // MeshStandardMaterial keeps the per-pixel cost low enough for the Quest.
  const material = new THREE.MeshStandardMaterial({
    color: tone,
    roughness: 0.55,
    metalness: 0.0,
    transparent: opacity < 1.0,
    opacity,
  });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.skinMap = { value: sharedTexture };
    shader.uniforms.skinScale = { value: scale };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSkinPos;\nvarying vec3 vSkinNormal;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSkinPos = position;\nvSkinNormal = normal;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D skinMap;\nuniform float skinScale;\nvarying vec3 vSkinPos;\nvarying vec3 vSkinNormal;')
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 w = pow(abs(normalize(vSkinNormal)), vec3(4.0));
          w /= (w.x + w.y + w.z + 1e-5);
          vec3 p = vSkinPos * skinScale;
          vec3 skin = texture2D(skinMap, p.yz).rgb * w.x
                    + texture2D(skinMap, p.xz).rgb * w.y
                    + texture2D(skinMap, p.xy).rgb * w.z;
          diffuseColor.rgb *= skin;
        }`);
  };
  material.customProgramCacheKey = () => 'superdex-skin';
  return material;
}
