import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_MCP = 9;

// Tracking constants — same audited pipeline as the AI app one level up.
const MAX_LOST_FRAMES = 25;
const JUMP_CONFIRM_FRAMES = 2;

// Effects run on a downscaled copy of the window region so the per-pixel
// passes stay real time no matter how big the frame gets on screen.
const FX_MAX = 320;

const cam = document.getElementById("cam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
// src holds the (optionally mirrored) camera frame; fx and tmp are the small
// scratch surfaces the per-pixel effects work on — those two get read back
// every frame, so they ask for CPU-backed storage.
const src = document.createElement("canvas");
const srcCtx = src.getContext("2d");
const fx = document.createElement("canvas");
const fxCtx = fx.getContext("2d", { willReadFrequently: true });
const tmp = document.createElement("canvas");
const tmpCtx = tmp.getContext("2d", { willReadFrequently: true });

const statusEl = document.getElementById("status");
const stage = document.getElementById("stage");
const btnStart = document.getElementById("btn-start");
const btnSnap = document.getElementById("btn-snap");
const btnRecord = document.getElementById("btn-record");
const mirrorBox = document.getElementById("mirror");
const fxBar = document.getElementById("fxbar");
const chipFps = document.getElementById("chip-fps");
const chipTrack = document.getElementById("chip-track");

let landmarker = null;
let stream = null;
let running = false;
let corners = null;
let presence = 0;
let frameActive = false;
let lostFrames = 0;
let jumpFrames = 0;
let lastTs = -1;
let recorder = null;
let recording = false;
let frameTimes = [];

function status(msg) {
  statusEl.textContent = msg;
  // Messages ending in an ellipsis are in-progress — show the spinner.
  statusEl.classList.toggle("working", /…\s*$/.test(msg));
}

// ---- effects (all local, all Canvas 2D — no network, no key) ----

// Deterministic per-cell noise: the same cell gets the same jitter on every
// frame, so brush strokes sit still instead of crawling.
function rand(n) {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const byte = (v) => (v < 0 ? 0 : v > 255 ? 255 : v | 0);

function luminance(data, w, h) {
  const L = new Float32Array(w * h);
  for (let p = 0, i = 0; p < L.length; p++, i += 4) {
    L[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return L;
}

// Sobel magnitude on a luminance plane (borders read as flat).
function sobel(L, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      const gx =
        -L[p - w - 1] - 2 * L[p - 1] - L[p + w - 1] +
        L[p - w + 1] + 2 * L[p + 1] + L[p + w + 1];
      const gy =
        -L[p - w - 1] - 2 * L[p - w] - L[p - w + 1] +
        L[p + w - 1] + 2 * L[p + w] + L[p + w + 1];
      out[p] = Math.hypot(gx, gy);
    }
  }
  return out;
}

function ramp(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = stops[0], b = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) {
        a = stops[s];
        b = stops[s + 1];
        break;
      }
    }
    const k = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[1][c] + (b[1][c] - a[1][c]) * k;
  }
  return lut;
}

const THERMAL = ramp([
  [0.0, [8, 0, 42]],
  [0.25, [72, 0, 150]],
  [0.5, [222, 40, 118]],
  [0.72, [255, 122, 24]],
  [0.88, [255, 214, 70]],
  [1.0, [255, 255, 236]],
]);

// Van Gogh: short curved strokes laid along the image's edge directions, with
// a slow swirl field filling the flat areas (the classic sky).
function fxVanGogh(c, w, h, t) {
  const img = c.getImageData(0, 0, w, h);
  const d = img.data;
  const L = luminance(d, w, h);

  c.save();
  c.filter = "saturate(1.4) contrast(1.06) brightness(0.94)";
  c.drawImage(c.canvas, 0, 0);
  c.filter = "none";
  c.lineCap = "round";

  const STEP = 7;
  const cols = Math.ceil(w / STEP);
  for (let gy = 0; gy < h; gy += STEP) {
    for (let gx = 0; gx < w; gx += STEP) {
      const id = (gy / STEP) * cols + gx / STEP;
      const x = clamp(gx + (rand(id) - 0.5) * STEP * 1.7, 1, w - 2);
      const y = clamp(gy + (rand(id + 7.7) - 0.5) * STEP * 1.7, 1, h - 2);
      const p = (y | 0) * w + (x | 0);
      const i = p * 4;

      const dx = L[p + 1] - L[p - 1];
      const dy = L[p + w] - L[p - w];
      const mag = Math.hypot(dx, dy);
      // Along the edge where there is one; a lazily rotating swirl where flat.
      const ang =
        mag > 9
          ? Math.atan2(dy, dx) + Math.PI / 2
          : Math.sin(x * 0.021) * 1.7 + Math.cos(y * 0.024) * 1.7 + t * 0.12;

      const len = STEP * 1.9 + rand(id + 3.1) * STEP;
      const cur = (rand(id + 11.3) - 0.5) * STEP * 1.2;
      const shade = 0.82 + rand(id + 5.5) * 0.42;
      const ux = Math.cos(ang), uy = Math.sin(ang);

      c.strokeStyle = `rgb(${byte(d[i] * shade)},${byte(d[i + 1] * shade)},${byte(d[i + 2] * shade)})`;
      c.lineWidth = 2 + rand(id + 2.2) * 1.9;
      c.beginPath();
      c.moveTo(x - ux * len * 0.5, y - uy * len * 0.5);
      c.quadraticCurveTo(x - uy * cur, y + ux * cur, x + ux * len * 0.5, y + uy * len * 0.5);
      c.stroke();
    }
  }
  c.restore();
}

// Cartoon: posterized colour with a hard ink outline.
function fxToon(c, w, h) {
  const img = c.getImageData(0, 0, w, h);
  const d = img.data;
  const edges = sobel(luminance(d, w, h), w, h);
  const LEVELS = 5;
  const q = 255 / (LEVELS - 1);
  for (let p = 0, i = 0; p < edges.length; p++, i += 4) {
    if (edges[p] > 95) {
      d[i] = 18;
      d[i + 1] = 16;
      d[i + 2] = 24;
    } else {
      d[i] = byte(Math.round(d[i] / q) * q * 1.05);
      d[i + 1] = byte(Math.round(d[i + 1] / q) * q * 1.05);
      d[i + 2] = byte(Math.round(d[i + 2] / q) * q * 1.05);
    }
  }
  c.putImageData(img, 0, 0);
}

// Neon: edges only, hue swept over time, then bloomed with a blur pass.
function fxNeon(c, w, h, t) {
  const img = c.getImageData(0, 0, w, h);
  const d = img.data;
  const edges = sobel(luminance(d, w, h), w, h);
  for (let p = 0, i = 0; p < edges.length; p++, i += 4) {
    const e = clamp(edges[p] / 210, 0, 1);
    const phase = ((p % w) / w + (p / w / h) * 0.5 + t * 0.12) * Math.PI * 2;
    d[i] = byte(e * (140 + 115 * Math.sin(phase)));
    d[i + 1] = byte(e * (120 + 110 * Math.sin(phase + 2.1)));
    d[i + 2] = byte(e * (170 + 85 * Math.sin(phase + 4.2)));
    d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  c.save();
  c.globalCompositeOperation = "lighter";
  c.filter = "blur(4px)";
  c.drawImage(c.canvas, 0, 0);
  c.restore();
}

// Glitch: RGB channel split, scanlines, and chunky band displacement that
// re-seeds a few times a second.
function fxGlitch(c, w, h, t) {
  const img = c.getImageData(0, 0, w, h);
  const d = img.data;
  const out = c.createImageData(w, h);
  const o = out.data;
  const k = 3 + Math.round(2 * Math.sin(t * 7));
  for (let y = 0; y < h; y++) {
    const dim = (y & 3) === 0 ? 0.7 : 1;
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const ir = (y * w + Math.min(w - 1, x + k)) * 4;
      const ib = (y * w + Math.max(0, x - k)) * 4;
      o[i] = d[ir] * dim;
      o[i + 1] = d[i + 1] * dim;
      o[i + 2] = d[ib + 2] * dim;
      o[i + 3] = 255;
    }
  }
  c.putImageData(out, 0, 0);

  const seed = Math.floor(t * 8);
  for (let b = 0; b < 5; b++) {
    const y = Math.floor(rand(seed * 13.7 + b) * h);
    const bh = 4 + Math.floor(rand(seed * 29.3 + b) * h * 0.12);
    const dx = (rand(seed * 7.9 + b) - 0.5) * w * 0.18;
    c.drawImage(c.canvas, 0, y, w, bh, dx, y, w, bh);
  }
}

// Pixel art: nearest-neighbour downsample, then posterize on the way back up.
function fxPixel(c, w, h) {
  const s = Math.max(2, Math.round(Math.min(w, h) / 26));
  const pw = Math.max(1, Math.round(w / s));
  const ph = Math.max(1, Math.round(h / s));
  tmp.width = pw;
  tmp.height = ph;
  tmpCtx.drawImage(c.canvas, 0, 0, w, h, 0, 0, pw, ph);
  const small = tmpCtx.getImageData(0, 0, pw, ph);
  const d = small.data;
  const q = 255 / 5;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = byte(Math.round(d[i] / q) * q);
    d[i + 1] = byte(Math.round(d[i + 1] / q) * q);
    d[i + 2] = byte(Math.round(d[i + 2] / q) * q);
  }
  tmpCtx.putImageData(small, 0, 0);
  c.imageSmoothingEnabled = false;
  c.clearRect(0, 0, w, h);
  c.drawImage(tmp, 0, 0, pw, ph, 0, 0, w, h);
  c.imageSmoothingEnabled = true;
}

// Thermal: luminance through a heat-map ramp.
function fxThermal(c, w, h) {
  const img = c.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
    d[i] = THERMAL[v * 3];
    d[i + 1] = THERMAL[v * 3 + 1];
    d[i + 2] = THERMAL[v * 3 + 2];
  }
  c.putImageData(img, 0, 0);
}

const EFFECTS = {
  vangogh: { name: "Van Gogh", render: fxVanGogh },
  toon: { name: "Cartoon", render: fxToon },
  neon: { name: "Neon", render: fxNeon },
  glitch: { name: "Glitch", render: fxGlitch },
  pixel: { name: "Pixel art", render: fxPixel, crisp: true },
  thermal: { name: "Thermal", render: fxThermal },
};
const FX_IDS = Object.keys(EFFECTS);
let effectId = localStorage.getItem("live-fx") || "vangogh";
if (!EFFECTS[effectId]) effectId = "vangogh";

function selectEffect(id) {
  if (!EFFECTS[id]) return;
  effectId = id;
  localStorage.setItem("live-fx", id);
  for (const b of fxBar.querySelectorAll("button")) {
    b.classList.toggle("on", b.dataset.fx === id);
  }
}

FX_IDS.forEach((id, i) => {
  const b = document.createElement("button");
  b.className = "pill";
  b.dataset.fx = id;
  b.innerHTML = `<span class="k">${i + 1}</span>${EFFECTS[id].name}`;
  b.addEventListener("click", () => selectEffect(id));
  fxBar.appendChild(b);
});
selectEffect(effectId);

mirrorBox.checked = localStorage.getItem("live-mirror") !== "0";
mirrorBox.addEventListener("change", () =>
  localStorage.setItem("live-mirror", mirrorBox.checked ? "1" : "0")
);

// ---- camera ----

async function initLandmarker() {
  if (landmarker) return;
  status("Loading hand tracker…");
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

async function startCamera() {
  btnStart.disabled = true;
  try {
    status("Asking for the camera…");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    cam.srcObject = stream;
    if (cam.readyState < 1) {
      await new Promise((res) => (cam.onloadedmetadata = res));
    }
    await cam.play();

    const w = cam.videoWidth || 1280;
    const h = cam.videoHeight || 720;
    canvas.width = src.width = w;
    canvas.height = src.height = h;
    stage.style.display = "flex";

    await initLandmarker();

    running = true;
    lastTs = -1;
    btnStart.textContent = "■ Stop camera";
    btnStart.classList.remove("primary");
    btnSnap.disabled = false;
    btnRecord.disabled = false;
    status("Make the two-hand finger frame — the effect lives inside it.");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    status(
      err?.name === "NotAllowedError"
        ? "⚠️ Camera permission denied — allow it in the browser and try again."
        : "⚠️ Could not start the camera: " + (err.message || err)
    );
  } finally {
    btnStart.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (recording) toggleRecording();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  cam.srcObject = null;
  corners = null;
  presence = 0;
  frameActive = false;
  btnStart.textContent = "▶ Start camera";
  btnStart.classList.add("primary");
  btnSnap.disabled = true;
  btnRecord.disabled = true;
  chipTrack.textContent = "camera off";
  chipFps.textContent = "— fps";
  status("Camera stopped.");
}

btnStart.addEventListener("click", () => (running ? stopCamera() : startCamera()));

// ---- tracking (ported from the AI app, which ports the live app) ----

function toPixel(lm) {
  return {
    x: (mirrorBox.checked ? 1 - lm.x : lm.x) * canvas.width,
    y: lm.y * canvas.height,
  };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function lerpPt(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}
function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a / 2);
}

function computeQuad(hands) {
  if (hands.length !== 2) return null;
  const info = hands.map((lm) => ({
    index: toPixel(lm[INDEX_TIP]),
    thumb: toPixel(lm[THUMB_TIP]),
    wristX: toPixel(lm[WRIST]).x,
    scale: dist(toPixel(lm[WRIST]), toPixel(lm[MIDDLE_MCP])) + 1,
  }));
  const needed = frameActive ? 0.2 : 0.75;
  for (const hd of info) {
    if (dist(hd.thumb, hd.index) < hd.scale * needed) return null;
  }
  info.sort((a, b) => a.wristX - b.wristX);
  const [A, B] = info;
  const pts = [A.index, B.index, B.thumb, A.thumb];
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  const hull = [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
  const minArea = frameActive ? 0.0005 : 0.005;
  if (polygonArea(hull) < canvas.width * canvas.height * minArea) return null;
  return pts;
}

function updateTracker(hands) {
  const target = computeQuad(hands);
  if (target) {
    if (!corners) {
      lostFrames = 0;
      frameActive = true;
      jumpFrames = 0;
      corners = target;
      presence = Math.min(1, presence + 0.12);
    } else {
      const moved = target.reduce((s, p, i) => s + dist(p, corners[i]), 0) / 4;
      if (moved > canvas.width * 0.3 && ++jumpFrames < JUMP_CONFIRM_FRAMES) {
        if (++lostFrames > MAX_LOST_FRAMES) presence = Math.max(0, presence - 0.05);
      } else {
        lostFrames = 0;
        frameActive = true;
        jumpFrames = 0;
        const alpha = Math.min(0.85, Math.max(0.35, moved / (canvas.width * 0.05)));
        corners = corners.map((c, i) => lerpPt(c, target[i], alpha));
        presence = Math.min(1, presence + 0.12);
      }
    }
  } else if (corners && ++lostFrames <= MAX_LOST_FRAMES) {
    presence = Math.min(1, presence + 0.12);
  } else {
    presence = Math.max(0, presence - 0.05);
    if (presence === 0) {
      corners = null;
      frameActive = false;
      jumpFrames = 0;
    }
  }
}

// ---- rendering ----

function quadPath(q) {
  ctx.beginPath();
  ctx.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(q[i].x, q[i].y);
  ctx.closePath();
}

function bboxOf(q, pad) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of q) {
    x0 = Math.min(x0, p.x);
    y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x);
    y1 = Math.max(y1, p.y);
  }
  x0 = clamp(Math.floor(x0 - pad), 0, canvas.width - 1);
  y0 = clamp(Math.floor(y0 - pad), 0, canvas.height - 1);
  x1 = clamp(Math.ceil(x1 + pad), x0 + 1, canvas.width);
  y1 = clamp(Math.ceil(y1 + pad), y0 + 1, canvas.height);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function drawWindow(q, t) {
  const bb = bboxOf(q, 6);
  const scale = Math.min(1, FX_MAX / Math.max(bb.w, bb.h));
  const fw = Math.max(2, Math.round(bb.w * scale));
  const fh = Math.max(2, Math.round(bb.h * scale));
  if (fx.width !== fw || fx.height !== fh) {
    fx.width = fw;
    fx.height = fh;
  }
  fxCtx.clearRect(0, 0, fw, fh);
  fxCtx.drawImage(src, bb.x, bb.y, bb.w, bb.h, 0, 0, fw, fh);

  const effect = EFFECTS[effectId];
  effect.render(fxCtx, fw, fh, t);

  ctx.save();
  quadPath(q);
  ctx.clip();
  ctx.globalAlpha = presence;
  ctx.imageSmoothingEnabled = !effect.crisp;
  ctx.drawImage(fx, 0, 0, fw, fh, bb.x, bb.y, bb.w, bb.h);
  ctx.restore();
}

function drawOutline(q, t) {
  ctx.save();
  ctx.globalAlpha = presence;
  quadPath(q);
  ctx.setLineDash([10, 8]);
  ctx.lineDashOffset = -t * 40;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;
  ctx.shadowBlur = 0;
  q.forEach((p, i) => {
    const r = 7 + Math.sin(t * 3 + i * 1.5) * 1.5;
    const halo = (t * 0.8 + i * 0.25) % 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + halo * 14, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${0.5 * (1 - halo) * presence})`;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (cam.readyState < 2 || !canvas.width) return;

  const now = performance.now();
  const t = now / 1000;

  srcCtx.save();
  if (mirrorBox.checked) {
    srcCtx.translate(canvas.width, 0);
    srcCtx.scale(-1, 1);
  }
  srcCtx.drawImage(cam, 0, 0, canvas.width, canvas.height);
  srcCtx.restore();
  ctx.drawImage(src, 0, 0);

  if (landmarker) {
    const ts = now > lastTs ? now : lastTs + 1;
    lastTs = ts;
    const res = landmarker.detectForVideo(cam, ts);
    updateTracker(res.landmarks || []);
  }

  if (corners && presence > 0.01) {
    drawWindow(corners, t);
    drawOutline(corners, t);
  }

  frameTimes.push(now);
  if (frameTimes.length > 30) frameTimes.shift();
  if (frameTimes.length > 1) {
    const span = frameTimes[frameTimes.length - 1] - frameTimes[0];
    chipFps.textContent = `${Math.round(((frameTimes.length - 1) * 1000) / span)} fps`;
  }
  chipTrack.textContent = frameActive
    ? "frame locked"
    : corners
    ? "holding…"
    : "show both hands";
}

// ---- capture ----

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

btnSnap.addEventListener("click", () => {
  canvas.toBlob((blob) => {
    if (!blob) return;
    download(blob, "finger-frame-live.png");
    status("Saved finger-frame-live.png");
  }, "image/png");
});

function toggleRecording() {
  if (recording) {
    recorder.stop();
    return;
  }
  const mime =
    [
      "video/mp4;codecs=avc1.42E01E",
      "video/mp4",
      "video/webm;codecs=vp9",
      "video/webm",
    ].find((m) => MediaRecorder.isTypeSupported(m)) || "video/webm";
  const isMp4 = mime.startsWith("video/mp4");
  recorder = new MediaRecorder(canvas.captureStream(30), {
    mimeType: mime,
    videoBitsPerSecond: 10_000_000,
  });
  const chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    recording = false;
    btnRecord.textContent = "● Record";
    btnRecord.classList.remove("rec");
    const ext = isMp4 ? "mp4" : "webm";
    download(new Blob(chunks, { type: isMp4 ? "video/mp4" : "video/webm" }), `finger-frame-live.${ext}`);
    status(
      `Saved finger-frame-live.${ext}.` +
        (isMp4
          ? ""
          : " (This browser records WebM — convert with: ffmpeg -i finger-frame-live.webm -c:v libx264 out.mp4)")
    );
  };
  recorder.start();
  recording = true;
  btnRecord.textContent = "■ Stop recording";
  btnRecord.classList.add("rec");
  status("Recording — press stop when you're done.");
}

btnRecord.addEventListener("click", toggleRecording);

// ---- keyboard ----
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= FX_IDS.length) return selectEffect(FX_IDS[n - 1]);
  const k = e.key.toLowerCase();
  if (k === "arrowright" || k === "arrowleft") {
    const i = FX_IDS.indexOf(effectId);
    const step = k === "arrowright" ? 1 : FX_IDS.length - 1;
    selectEffect(FX_IDS[(i + step) % FX_IDS.length]);
  } else if (k === "m") {
    mirrorBox.checked = !mirrorBox.checked;
    mirrorBox.dispatchEvent(new Event("change"));
  } else if (k === "s" && !btnSnap.disabled) {
    btnSnap.click();
  } else if (k === "r" && !btnRecord.disabled) {
    toggleRecording();
  }
});

// Dev hook: render one frame of a given effect through a fixed window quad,
// using whatever the camera last painted — lets automated checks exercise the
// effect pipeline (and time it) without a pair of hands in front of the lens.
// Returns the render time in milliseconds.
window.__forceFrame = (id) => {
  selectEffect(id);
  const w = canvas.width, h = canvas.height;
  const q = [
    { x: w * 0.3, y: h * 0.28 },
    { x: w * 0.72, y: h * 0.28 },
    { x: w * 0.72, y: h * 0.74 },
    { x: w * 0.3, y: h * 0.74 },
  ];
  presence = 1;
  const t0 = performance.now();
  drawWindow(q, 1.5);
  drawOutline(q, 1.5);
  return +(performance.now() - t0).toFixed(2);
};

status("Start the camera to begin — nothing leaves your device.");
