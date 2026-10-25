import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = (id) => document.getElementById(id);
const video = $("webcam");
const stage = $("stage");
const drawCanvas = $("drawingCanvas");
const cursorCanvas = $("cursorCanvas");
const drawCtx = drawCanvas.getContext("2d", { alpha: true });
const cursorCtx = cursorCanvas.getContext("2d");
const overlay = $("overlay");

const state = {
  landmarker: null, stream: null, lastVideoTime: -1, raf: 0,
  color: "#f7f7fb", brushSize: 8, mode: "idle", candidate: "idle", candidateFrames: 0,
  lastPoint: null, smoothPoint: null, lastRaw: null, lastTimestamp: 0,
  strokes: [], currentStroke: null, dpr: 1, width: 0, height: 0,
  lastHandSeen: 0, handLostTimer: null, enabled: false
};

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function setDot(id, kind) {
  const dot = $(id);
  dot.classList.remove("good", "bad");
  if (kind) dot.classList.add(kind);
}
function setStatus(id, text) { $(id).textContent = text; }

function resizeCanvases() {
  const rect = stage.getBoundingClientRect();
  state.dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.width = rect.width;
  state.height = rect.height;
  for (const canvas of [drawCanvas, cursorCanvas]) {
    canvas.width = Math.round(rect.width * state.dpr);
    canvas.height = Math.round(rect.height * state.dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  }
  redrawAll();
}
function mapPoint(point) {
  // The video and canvas share a 16:9 stage. Mirroring matches the webcam preview.
  return { x: (1 - point.x) * state.width, y: point.y * state.height };
}
function normalizedPoint(p) { return { x: p.x, y: p.y }; }

function drawSmoothSegment(ctx, from, to, color, width, erase = false) {
  const a = mapPoint(from), b = mapPoint(to);
  const distancePx = distance(a, b);
  if (distancePx < 0.15) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  if (erase) ctx.globalCompositeOperation = "destination-out";

  // Subdivide fast movements so a quick hand motion never leaves gaps.
  const steps = Math.max(1, Math.ceil(distancePx / Math.max(2, width * 0.35)));
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    ctx.lineTo(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
  }
  ctx.stroke();
  ctx.restore();
}
function drawDot(ctx, point, color, width, erase = false) {
  const p = mapPoint(point);
  ctx.save();
  if (erase) ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, width / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
function redrawAll() {
  if (!state.width || !state.height) return;
  drawCtx.clearRect(0, 0, state.width, state.height);
  for (const stroke of state.strokes) {
    if (!stroke.points.length) continue;
    const erasing = stroke.type === "eraser";
    if (stroke.points.length === 1) {
      drawDot(drawCtx, stroke.points[0], stroke.color || "#000", stroke.width, erasing);
      continue;
    }
    for (let i = 1; i < stroke.points.length; i++) {
      drawSmoothSegment(drawCtx, stroke.points[i - 1], stroke.points[i], stroke.color || "#000", stroke.width, erasing);
    }
  }
}
function finishStroke() {
  state.currentStroke = null;
  state.lastPoint = null;
}
function addDrawPoint(point) {
  const type = state.mode === "eraser" ? "eraser" : "pen";
  const width = type === "eraser" ? Math.max(26, state.brushSize * 4.5) : state.brushSize;
  if (!state.currentStroke || state.currentStroke.type !== type) {
    finishStroke();
    state.currentStroke = { type, color: state.color, width, points: [normalizedPoint(point)] };
    state.strokes.push(state.currentStroke);
    drawDot(drawCtx, point, state.color, width, type === "eraser");
    state.lastPoint = normalizedPoint(point);
    return;
  }
  const previous = state.lastPoint;
  if (!previous) {
    state.lastPoint = normalizedPoint(point);
    return;
  }
  const a = mapPoint(previous), b = mapPoint(point);
  if (distance(a, b) < 0.6) return;
  const next = normalizedPoint(point);
  state.currentStroke.points.push(next);
  drawSmoothSegment(drawCtx, previous, next, state.color, width, type === "eraser");
  state.lastPoint = next;
}

function angle(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
  const dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (!m1 || !m2) return 0;
  return Math.acos(clamp(dot / (m1 * m2), -1, 1)) * 180 / Math.PI;
}
function fingerExtended(lm, mcp, pip, tip) {
  return angle(lm[mcp], lm[pip], lm[tip]) > 148;
}
function classifyGesture(lm) {
  const index = fingerExtended(lm, 5, 6, 8);
  const middle = fingerExtended(lm, 9, 10, 12);
  const ring = fingerExtended(lm, 13, 14, 16);
  const pinky = fingerExtended(lm, 17, 18, 20);
  const count = [index, middle, ring, pinky].filter(Boolean).length;
  if (index && !middle && !ring && !pinky) return "pen";
  if (!index && !middle && !ring && !pinky) return "eraser";
  if (count >= 3) return "idle";
  return "uncertain";
}
function commitGesture(next) {
  if (next === "uncertain") {
    state.candidateFrames = 0;
    return state.mode;
  }
  if (next === state.candidate) state.candidateFrames++;
  else { state.candidate = next; state.candidateFrames = 1; }
  // Require a few stable frames, but don't let a short transition break the stroke.
  if (state.candidateFrames >= 3 && state.mode !== next) {
    state.mode = next;
    finishStroke();
  }
  return state.mode;
}
function smoothPoint(raw, now) {
  if (!state.smoothPoint || !state.lastRaw) {
    state.smoothPoint = { x: raw.x, y: raw.y };
    state.lastRaw = { x: raw.x, y: raw.y };
    state.lastTimestamp = now;
    return state.smoothPoint;
  }
  const dt = Math.max(1, now - state.lastTimestamp);
  const speed = distance(raw, state.lastRaw) / dt;
  // More filtering when still; less filtering during quick movement to avoid lag.
  const alpha = clamp(0.24 + speed * 4.2, 0.24, 0.78);
  state.smoothPoint.x += alpha * (raw.x - state.smoothPoint.x);
  state.smoothPoint.y += alpha * (raw.y - state.smoothPoint.y);
  state.lastRaw = { x: raw.x, y: raw.y };
  state.lastTimestamp = now;
  return state.smoothPoint;
}
function palmCenter(lm) {
  const ids = [0, 5, 9, 13, 17];
  return ids.reduce((p, id) => ({ x: p.x + lm[id].x / ids.length, y: p.y + lm[id].y / ids.length }), { x: 0, y: 0 });
}
function drawCursor(point, mode) {
  cursorCtx.clearRect(0, 0, state.width, state.height);
  const p = mapPoint(point);
  cursorCtx.save();
  cursorCtx.beginPath();
  cursorCtx.arc(p.x, p.y, mode === "eraser" ? Math.max(13, state.brushSize * 2.25) : Math.max(5, state.brushSize / 2), 0, Math.PI * 2);
  cursorCtx.strokeStyle = mode === "eraser" ? "#fb7185" : state.color;
  cursorCtx.lineWidth = 1.5;
  cursorCtx.stroke();
  cursorCtx.beginPath();
  cursorCtx.arc(p.x, p.y, 2, 0, Math.PI * 2);
  cursorCtx.fillStyle = "#ffffff";
  cursorCtx.fill();
  cursorCtx.restore();
}
function clearCursor() { cursorCtx.clearRect(0, 0, state.width, state.height); }

async function startCamera() {
  $("overlayTitle").textContent = "Getting things ready";
  $("overlayMessage").textContent = "Allow camera access, then start drawing with your hand.";
  $("startBtn").disabled = true;
  $("startBtn").textContent = "Starting camera…";
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = state.stream;
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = reject;
    });
    await video.play();
    setStatus("modelStatus", "LOADING MODEL");
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    try {
      state.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45
      });
    } catch (gpuError) {
      state.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45
      });
    }
    resizeCanvases();
    window.addEventListener("resize", resizeCanvases);
    $("modelDot").classList.add("good");
    setStatus("modelStatus", "MODEL READY");
    setStatus("gestureStatus", "SHOW YOUR HAND");
    overlay.classList.add("hidden");
    state.enabled = true;
    requestAnimationFrame(processFrame);
  } catch (err) {
    console.error(err);
    $("overlayTitle").textContent = "Could not start Air Canvas";
    $("overlayMessage").textContent = err?.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser, then retry."
      : `Could not initialize the camera or hand model. ${err?.message || ""}`;
    $("startBtn").disabled = false;
    $("startBtn").textContent = "Try again";
    setStatus("modelStatus", "MODEL ERROR");
    setDot("modelDot", "bad");
  }
}

function processFrame(now) {
  if (!state.enabled || !state.landmarker) return;
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    let results;
    try {
      results = state.landmarker.detectForVideo(video, now);
    } catch (err) {
      console.warn("Hand tracking frame skipped:", err);
      requestAnimationFrame(processFrame);
      return;
    }
    const landmarks = results?.landmarks?.[0];
    if (landmarks) {
      state.lastHandSeen = now;
      $("stageHint").classList.add("hidden");
      setDot("handDot", "good");
      setStatus("handStatus", "HAND DETECTED");
      const detected = classifyGesture(landmarks);
      const mode = commitGesture(detected);
      const tracking = mode === "eraser" ? palmCenter(landmarks) : landmarks[8];
      const point = smoothPoint(tracking, now);
      drawCursor(point, mode);
      if (mode === "pen" || mode === "eraser") {
        addDrawPoint(point);
        setStatus("gestureStatus", mode === "pen" ? "DRAWING" : "ERASING");
        $("modeText").textContent = mode === "pen" ? "Drawing with index finger" : "Erasing with fist";
      } else {
        finishStroke();
        setStatus("gestureStatus", "PAUSED");
        $("modeText").textContent = "Open palm — drawing paused";
      }
    } else if (now - state.lastHandSeen > 180) {
      setDot("handDot", "bad");
      setStatus("handStatus", "HAND NOT FOUND");
      setStatus("gestureStatus", "SHOW YOUR HAND");
      $("modeText").textContent = "Waiting for hand";
      $("stageHint").classList.remove("hidden");
      finishStroke();
      state.smoothPoint = null;
      state.lastRaw = null;
      clearCursor();
    }
  }
  state.raf = requestAnimationFrame(processFrame);
}

function setupControls() {
  document.querySelectorAll(".swatch").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".swatch").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      state.color = btn.dataset.color;
      finishStroke();
    });
  });
  $("brushSize").addEventListener("input", e => {
    state.brushSize = Number(e.target.value);
    $("brushValue").textContent = `${state.brushSize} px`;
  });
  $("clearBtn").addEventListener("click", () => {
    state.strokes = [];
    finishStroke();
    drawCtx.clearRect(0, 0, state.width, state.height);
  });
  $("undoBtn").addEventListener("click", () => {
    state.strokes.pop();
    finishStroke();
    redrawAll();
  });
  $("saveBtn").addEventListener("click", () => {
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = drawCanvas.width;
    exportCanvas.height = drawCanvas.height;
    exportCanvas.getContext("2d").drawImage(drawCanvas, 0, 0);
    const link = document.createElement("a");
    link.download = `air-canvas-${new Date().toISOString().slice(0,10)}.png`;
    link.href = exportCanvas.toDataURL("image/png");
    link.click();
  });
  $("startBtn").addEventListener("click", startCamera);
  document.addEventListener("keydown", e => {
    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    const key = e.key.toLowerCase();
    if (key === "c") $("clearBtn").click();
    if (key === "u") $("undoBtn").click();
    if (key === "s") $("saveBtn").click();
  });
}
setupControls();
window.addEventListener("beforeunload", () => {
  state.enabled = false;
  if (state.raf) cancelAnimationFrame(state.raf);
  state.stream?.getTracks().forEach(track => track.stop());
  state.landmarker?.close();
});
