// Copyright (c) Meta Platforms, Inc. and affiliates.
// Licensed under the Apache License, Version 2.0.
//
// A procedural kitchen around the teleoperation table (display only; nothing
// here takes part in the physics). The kitchen's origin is the floor point
// under the table center, with -Z pointing away from the operator. The
// physics table itself is drawn as the top of a kitchen island (see
// makeIsland), so the manipulated objects sit on it.

import * as THREE from 'three';

// --------------------------------------------------------------------------
// Canvas textures
// --------------------------------------------------------------------------

function canvasTexture(width, height, draw, repeat = [1, 1]) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d'), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeat[0], repeat[1]);
  texture.anisotropy = 4;
  return texture;
}

// Deterministic pseudo-random numbers so the kitchen looks the same each load.
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function woodFloor() {
  return canvasTexture(1024, 1024, (ctx, w, h) => {
    const r = rng(7);
    const plank = h / 8;
    for (let row = 0; row < 8; row++) {
      let x = -r() * 400;
      while (x < w) {
        const len = 300 + r() * 380;
        const tone = 118 + r() * 34;
        ctx.fillStyle = `rgb(${tone + 38},${tone + 4},${tone - 40})`;
        ctx.fillRect(x, row * plank, len, plank);
        for (let g = 0; g < 14; g++) {  // grain
          ctx.strokeStyle = `rgba(70,40,15,${0.05 + r() * 0.08})`;
          ctx.lineWidth = 1 + r() * 2;
          const y = row * plank + r() * plank;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.bezierCurveTo(x + len * 0.3, y + (r() - 0.5) * 10, x + len * 0.7, y + (r() - 0.5) * 10, x + len, y);
          ctx.stroke();
        }
        ctx.strokeStyle = 'rgba(40,24,10,0.55)';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, row * plank, len, plank);
        x += len;
      }
    }
  }, [4, 4]);
}

function marble(base = [236, 234, 230]) {
  return canvasTexture(1024, 1024, (ctx, w, h) => {
    const r = rng(11);
    ctx.fillStyle = `rgb(${base.join(',')})`;
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 26; i++) {
      ctx.strokeStyle = `rgba(120,120,128,${0.08 + r() * 0.22})`;
      ctx.lineWidth = 0.6 + r() * 2.2;
      ctx.beginPath();
      let x = r() * w, y = r() * h;
      ctx.moveTo(x, y);
      for (let k = 0; k < 9; k++) {
        x += (r() - 0.3) * 150;
        y += (r() - 0.5) * 120;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [1, 1]);
}

function butcherBlock() {
  return canvasTexture(1024, 512, (ctx, w, h) => {
    const r = rng(3);
    const strip = w / 24;
    for (let i = 0; i < 24; i++) {
      const t = 150 + r() * 45;
      ctx.fillStyle = `rgb(${t + 40},${t},${t - 55})`;
      ctx.fillRect(i * strip, 0, strip, h);
      for (let g = 0; g < 6; g++) {
        ctx.strokeStyle = `rgba(90,55,20,${0.06 + r() * 0.1})`;
        ctx.lineWidth = 1;
        const x = i * strip + r() * strip;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x + (r() - 0.5) * 6, h); ctx.stroke();
      }
      ctx.fillStyle = 'rgba(60,35,10,0.35)';
      ctx.fillRect(i * strip, 0, 1.5, h);
    }
  }, [1, 1]);
}

function subwayTiles() {
  return canvasTexture(512, 512, (ctx, w, h) => {
    const r = rng(5);
    ctx.fillStyle = '#c9ccce';  // grout
    ctx.fillRect(0, 0, w, h);
    const tw = w / 4, th = h / 8;
    for (let row = 0; row < 8; row++) {
      const offset = (row % 2) * tw / 2;
      for (let col = -1; col < 5; col++) {
        const x = col * tw + offset, y = row * th;
        const t = 238 + r() * 12;
        const grad = ctx.createLinearGradient(x, y, x, y + th);
        grad.addColorStop(0, `rgb(${t},${t},${t - 2})`);
        grad.addColorStop(1, `rgb(${t - 14},${t - 12},${t - 12})`);
        ctx.fillStyle = grad;
        ctx.fillRect(x + 3, y + 3, tw - 6, th - 6);
      }
    }
  }, [6, 2]);
}

function cabinetDoors(color = '#5f7d73', doors = 4, drawer = false) {
  return canvasTexture(1024, 512, (ctx, w, h) => {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, w, h);
    const dw = w / doors;
    for (let i = 0; i < doors; i++) {
      const x = i * dw;
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 6;
      ctx.strokeRect(x + 6, 6, dw - 12, h - 12);           // door gap
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 10;
      ctx.strokeRect(x + 34, 40, dw - 68, h - 80);         // shaker panel
      ctx.fillStyle = '#c9c4b8';                            // brass handle
      if (drawer) ctx.fillRect(x + dw / 2 - 40, 26, 80, 12);
      else ctx.fillRect(i % 2 ? x + 20 : x + dw - 30, h / 2 - 40, 10, 80);
    }
  }, [1, 1]);
}

function paint(color) {
  return canvasTexture(256, 256, (ctx, w, h) => {
    const r = rng(9);
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 1800; i++) {
      ctx.fillStyle = `rgba(${r() > 0.5 ? 255 : 0},${r() > 0.5 ? 255 : 0},${r() > 0.5 ? 255 : 0},0.025)`;
      ctx.fillRect(r() * w, r() * h, 2, 2);
    }
  }, [6, 3]);
}

function windowView() {
  return canvasTexture(512, 512, (ctx, w, h) => {
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, '#8fc3ee');
    sky.addColorStop(0.65, '#dcefff');
    sky.addColorStop(0.66, '#6f9a5b');
    sky.addColorStop(1, '#4f7a41');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
    const r = rng(21);
    for (let i = 0; i < 9; i++) {  // trees
      ctx.fillStyle = `rgb(${50 + r() * 30},${100 + r() * 40},${50 + r() * 20})`;
      ctx.beginPath();
      ctx.arc(r() * w, h * 0.62, 30 + r() * 50, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.ellipse(r() * w, 60 + r() * 120, 60, 18, 0, 0, Math.PI * 2); ctx.fill(); }
  });
}

// --------------------------------------------------------------------------
// Geometry helpers
// --------------------------------------------------------------------------

function box(parent, size, position, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...position);
  parent.add(mesh);
  return mesh;
}

function cylinder(parent, radiusTop, radiusBottom, height, position, material, segments = 24) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radiusTop, radiusBottom, height, segments), material);
  mesh.position.set(...position);
  parent.add(mesh);
  return mesh;
}

// --------------------------------------------------------------------------
// Kitchen
// --------------------------------------------------------------------------

const ROOM = { halfWidth: 2.3, back: -1.7, front: 2.4, height: 2.7 };
const COUNTER = { height: 0.9, depth: 0.62 };

export function buildKitchen() {
  const group = new THREE.Group();
  group.name = 'kitchen';

  const mats = {
    floor: new THREE.MeshStandardMaterial({ map: woodFloor(), roughness: 0.7 }),
    wall: new THREE.MeshStandardMaterial({ map: paint('#e9e3d6'), roughness: 0.95 }),
    ceiling: new THREE.MeshStandardMaterial({ color: 0xf6f4ef, roughness: 1.0 }),
    tiles: new THREE.MeshStandardMaterial({ map: subwayTiles(), roughness: 0.25 }),
    counter: new THREE.MeshStandardMaterial({ map: marble(), roughness: 0.25 }),
    base: new THREE.MeshStandardMaterial({ map: cabinetDoors('#5f7d73', 4), roughness: 0.6 }),
    upper: new THREE.MeshStandardMaterial({ map: cabinetDoors('#e8e4da', 3), roughness: 0.6 }),
    side: new THREE.MeshStandardMaterial({ color: 0x5f7d73, roughness: 0.6 }),
    upperSide: new THREE.MeshStandardMaterial({ color: 0xe8e4da, roughness: 0.6 }),
    steel: new THREE.MeshStandardMaterial({ color: 0xc8ccd0, metalness: 0.85, roughness: 0.28 }),
    dark: new THREE.MeshStandardMaterial({ color: 0x1b1c1f, metalness: 0.2, roughness: 0.25 }),
    glass: new THREE.MeshStandardMaterial({ map: windowView(), emissive: 0xffffff, emissiveIntensity: 0.55, roughness: 0.1 }),
    frame: new THREE.MeshStandardMaterial({ color: 0xf4f2ec, roughness: 0.5 }),
    wood: new THREE.MeshStandardMaterial({ map: butcherBlock(), roughness: 0.55 }),
    ceramic: new THREE.MeshStandardMaterial({ color: 0xf2efe8, roughness: 0.3 }),
    leaf: new THREE.MeshStandardMaterial({ color: 0x3f7d3a, roughness: 0.8 }),
    terracotta: new THREE.MeshStandardMaterial({ color: 0xb4623e, roughness: 0.9 }),
  };

  const room = new THREE.Group();
  room.name = 'kitchen-room';
  group.add(room);
  const W = ROOM.halfWidth * 2, D = ROOM.front - ROOM.back, H = ROOM.height, zc = (ROOM.front + ROOM.back) / 2;

  // Floor, ceiling, walls (inward facing).
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(W, D), mats.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, zc);
  room.add(floor);
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(W, D), mats.ceiling);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, H, zc);
  room.add(ceiling);
  const walls = [
    [W, [0, H / 2, ROOM.back], 0],
    [W, [0, H / 2, ROOM.front], Math.PI],
    [D, [-ROOM.halfWidth, H / 2, zc], Math.PI / 2],
    [D, [ROOM.halfWidth, H / 2, zc], -Math.PI / 2],
  ];
  for (const [width, pos, yaw] of walls) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(width, H), mats.wall);
    wall.position.set(...pos);
    wall.rotation.y = yaw;
    room.add(wall);
  }

  // Back counter run with sink, stove and upper cabinets.
  const zBack = ROOM.back + COUNTER.depth / 2;
  const runWidth = 3.4;
  box(room, [runWidth, COUNTER.height - 0.04, COUNTER.depth - 0.02], [0, (COUNTER.height - 0.04) / 2, zBack + 0.01], [
    mats.side, mats.side, mats.side, mats.side, mats.base, mats.side]);
  box(room, [runWidth + 0.04, 0.04, COUNTER.depth + 0.03], [0, COUNTER.height - 0.02, zBack + 0.015], mats.counter);
  const splash = new THREE.Mesh(new THREE.PlaneGeometry(runWidth, 0.55), mats.tiles);
  splash.position.set(0, COUNTER.height + 0.275, ROOM.back + 0.005);
  room.add(splash);
  for (const x of [-1.15, 1.15]) {
    box(room, [1.1, 0.72, 0.36], [x, 1.86, ROOM.back + 0.18], [
      mats.upperSide, mats.upperSide, mats.upperSide, mats.upperSide, mats.upper, mats.upperSide]);
  }
  // Window above the sink.
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.8), mats.glass);
  glass.position.set(0, 1.85, ROOM.back + 0.01);
  room.add(glass);
  for (const [w, h, x, y] of [[1.08, 0.05, 0, 2.27], [1.08, 0.05, 0, 1.43], [0.05, 0.86, -0.52, 1.85], [0.05, 0.86, 0.52, 1.85], [0.03, 0.8, 0, 1.85]]) {
    box(room, [w, h, 0.05], [x, y, ROOM.back + 0.03], mats.frame);
  }
  // Sink and faucet.
  box(room, [0.6, 0.012, 0.42], [0, COUNTER.height + 0.001, zBack + 0.03], mats.steel);
  box(room, [0.52, 0.01, 0.34], [0, COUNTER.height + 0.008, zBack + 0.03], mats.dark);
  cylinder(room, 0.014, 0.018, 0.3, [0, COUNTER.height + 0.15, ROOM.back + 0.1], mats.steel);
  const spout = new THREE.Mesh(new THREE.TorusGeometry(0.08, 0.012, 10, 24, Math.PI), mats.steel);
  spout.position.set(0, COUNTER.height + 0.3, ROOM.back + 0.18);
  spout.rotation.y = Math.PI / 2;
  room.add(spout);
  // Cooktop with burners, and a range hood.
  box(room, [0.62, 0.012, 0.52], [1.15, COUNTER.height + 0.006, zBack], mats.dark);
  for (const [dx, dz, rr] of [[-0.15, -0.12, 0.09], [0.15, -0.12, 0.07], [-0.15, 0.12, 0.07], [0.15, 0.12, 0.09]]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(rr, 0.004, 6, 40),
      new THREE.MeshStandardMaterial({ color: 0x9a3b2a, emissive: 0x3a0c05, roughness: 0.4 }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(1.15 + dx, COUNTER.height + 0.013, zBack + dz);
    room.add(ring);
  }
  box(room, [0.7, 0.28, 0.45], [1.15, 1.95, ROOM.back + 0.225], mats.steel);
  box(room, [0.28, 0.5, 0.25], [1.15, 2.42, ROOM.back + 0.125], mats.steel);

  // Fridge in the left corner.
  const fridgeX = -ROOM.halfWidth + 0.42;
  box(room, [0.8, 1.9, 0.72], [fridgeX, 0.95, ROOM.back + 0.38], mats.steel);
  box(room, [0.8, 0.008, 0.73], [fridgeX, 1.25, ROOM.back + 0.38], mats.dark);
  for (const y of [0.75, 1.6]) box(room, [0.03, 0.4, 0.03], [fridgeX + 0.32, y, ROOM.back + 0.76], mats.steel);

  // Open shelves on the right wall with jars and plates.
  for (const y of [1.35, 1.7]) {
    box(room, [0.28, 0.03, 1.2], [ROOM.halfWidth - 0.14, y, -0.4], mats.wood);
    for (let i = 0; i < 5; i++) {
      const jar = cylinder(room, 0.045, 0.045, 0.14, [ROOM.halfWidth - 0.14, y + 0.085, -0.9 + i * 0.24],
        new THREE.MeshStandardMaterial({ color: [0xd9b77a, 0xa15b3b, 0xe8d8a8, 0x7b8f4a, 0xc4b09a][i], roughness: 0.4 }));
      jar.userData.decor = true;
    }
  }

  // Counter props: cutting board, fruit bowl, kettle and a plant.
  box(room, [0.4, 0.02, 0.26], [-0.8, COUNTER.height + 0.01, zBack + 0.05], mats.wood);
  const bowl = new THREE.Mesh(new THREE.SphereGeometry(0.14, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), mats.ceramic);
  bowl.position.set(0.55, COUNTER.height + 0.14, zBack + 0.02);
  room.add(bowl);
  const fruit = [0xd23a2a, 0xf2b01e, 0x7cb342, 0xe0662a, 0xc62828];
  fruit.forEach((color, i) => {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 12), new THREE.MeshStandardMaterial({ color, roughness: 0.45 }));
    s.position.set(0.55 + Math.cos(i * 1.3) * 0.06, COUNTER.height + 0.05 + (i === 4 ? 0.05 : 0), zBack + 0.02 + Math.sin(i * 1.3) * 0.06);
    room.add(s);
  });
  const kettle = new THREE.Mesh(new THREE.SphereGeometry(0.09, 24, 16), mats.steel);
  kettle.scale.set(1, 0.9, 1);
  kettle.position.set(1.0, COUNTER.height + 0.1, zBack + 0.1);
  room.add(kettle);
  cylinder(room, 0.07, 0.055, 0.12, [-1.45, COUNTER.height + 0.06, zBack - 0.05], mats.terracotta);
  for (let i = 0; i < 7; i++) {
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), mats.leaf);
    leaf.scale.set(0.6, 1.4, 0.35);
    leaf.position.set(-1.45 + Math.cos(i) * 0.05, COUNTER.height + 0.19 + (i % 3) * 0.03, zBack - 0.05 + Math.sin(i) * 0.05);
    leaf.rotation.z = Math.cos(i * 2) * 0.6;
    room.add(leaf);
  }

  // Ceiling light panel and pendant lamps over the island.
  const panel = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.6),
    new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff4e0, emissiveIntensity: 1.2 }));
  panel.rotation.x = Math.PI / 2;
  panel.position.set(0, H - 0.01, 0.6);
  room.add(panel);
  const shade = new THREE.MeshStandardMaterial({ color: 0x23262b, roughness: 0.4, metalness: 0.4, side: THREE.DoubleSide });
  const bulb = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffe2b0, emissiveIntensity: 2.0 });
  for (const x of [-0.35, 0.35]) {
    cylinder(room, 0.004, 0.004, H - 2.05, [x, (H + 2.05) / 2, 0], shade, 6);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.16, 24, 1, true), shade);
    cone.position.set(x, 2.02, 0);
    room.add(cone);
    const b = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), bulb);
    b.position.set(x, 1.96, 0);
    room.add(b);
  }
  const lamp = new THREE.PointLight(0xffe7c4, 3.0, 4.0, 1.6);
  lamp.position.set(0, 1.9, 0);
  room.add(lamp);

  return { group, room };
}

// The table the physics runs on, drawn as a butcher-block island top. Its
// top surface is the physics frame's y = 0; the cabinet body below reaches
// down to the floor (call setHeight with the table height above the floor).
export function makeIsland(width = 1.2, depth = 0.8) {
  const group = new THREE.Group();
  const top = new THREE.Mesh(new THREE.BoxGeometry(width, 0.04, depth),
    new THREE.MeshStandardMaterial({ map: butcherBlock(), roughness: 0.55 }));
  top.position.y = -0.02;
  group.add(top);
  const doors = new THREE.MeshStandardMaterial({ map: cabinetDoors('#5f7d73', 3), roughness: 0.6 });
  const side = new THREE.MeshStandardMaterial({ color: 0x5f7d73, roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(width - 0.08, 1, depth - 0.1), [side, side, side, side, doors, doors]);
  group.add(body);
  const kick = new THREE.Mesh(new THREE.BoxGeometry(width - 0.16, 0.08, depth - 0.18),
    new THREE.MeshStandardMaterial({ color: 0x2c3431, roughness: 0.8 }));
  group.add(kick);
  function setHeight(height) {
    const h = Math.max(0.2, height - 0.04);
    body.scale.y = h - 0.08;
    body.position.y = -0.04 - (h - 0.08) / 2;
    kick.position.y = -height + 0.04;
    body.visible = kick.visible = height > 0.25;
  }
  setHeight(COUNTER.height);
  return { group, top, setHeight };
}
