/* Finger Count — MediaPipe finds the hand, a YOLO11 ONNX classifier counts the fingers.
   Everything runs in the browser: no uploads, no server. */
(() => {
  // Vendored first so the site works offline / behind a firewall; CDN is the fallback.
  const ORT_LOCAL = "vendor/ort/ort.min.js";
  const ORT_CDN = [
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js",
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js",
  ];
  const ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  // absolute URLs: a bare "vendor/..." specifier is not resolvable by import()
  const MP_LOCAL = new URL("vendor/mp/", location.href).href;
  const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/";
  const HAND_MODEL_LOCAL = new URL("vendor/mp/hand_landmarker.task", location.href).href;
  const HAND_MODEL_CDN =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
  const MODEL_PATH = "models/finger_count.onnx";
  const SIZE = 224;
  const PAD = 0.25;

  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const photo = $("photo");
  const overlay = $("overlay");
  const placeholder = $("placeholder");
  const statusEl = $("status");
  const countValue = $("count-value");
  const countLabel = $("count-label");
  const confFill = $("conf-fill");
  const confText = $("conf-text");
  const fpsEl = $("fps");

  let session = null;
  let landmarker = null;
  let vision = null;
  let running = false;
  let mode = "idle"; // idle | camera | photo
  let lastVideoTime = -1;
  let frames = 0;
  let fpsStart = performance.now();

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = cropCanvas.height = SIZE;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

  const setStatus = (t) => (statusEl.textContent = t);

  function loadScript(urls) {
    return new Promise((resolve, reject) => {
      let i = 0;
      const next = () => {
        if (i >= urls.length) return reject(new Error("could not load onnxruntime-web"));
        const s = document.createElement("script");
        s.src = urls[i++];
        s.onload = () => resolve();
        s.onerror = next;
        document.head.appendChild(s);
      };
      next();
    });
  }

  async function init() {
    if (session && landmarker) return;
    setStatus("loading runtime…");
    if (!window.ort) {
      try {
        await loadScript([ORT_LOCAL]);
      } catch (_) {
        /* fall through to CDN */
      }
    }
    ort.env.wasm.numThreads = 1;   // no SharedArrayBuffer without COOP/COEP headers
    if (window.ort) {
      ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
    } else {
      await loadScript(ORT_CDN);
      if (!window.ort) throw new Error("onnxruntime-web failed to load (local and CDN)");
      ort.env.wasm.wasmPaths = ORT_CDN_BASE;
    }

    setStatus("loading model…");
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    setStatus("loading hand detector…");
    let mpBase = MP_LOCAL, modelPath = HAND_MODEL_LOCAL;
    try {
      vision = await import(`${MP_LOCAL}vision_bundle.mjs`);
    } catch (e1) {
      mpBase = MP_BASE;
      modelPath = HAND_MODEL_CDN;
      try {
        vision = await import(`${MP_BASE}vision_bundle.mjs`);
      } catch (e2) {
        throw new Error(`local[${e1.message}] cdn[${e2.message}]`);
      }
    }
    const fileset = await vision.FilesetResolver.forVisionTasks(`${mpBase}wasm`);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: modelPath, delegate },
      runningMode: "VIDEO",
      numHands: 1,
    });
    try {
      landmarker = await vision.HandLandmarker.createFromOptions(fileset, opts("GPU"));
    } catch (e) {
      landmarker = await vision.HandLandmarker.createFromOptions(fileset, opts("CPU"));
    }
    setStatus("ready — hold one hand up");
  }

  /* ---------- geometry ---------- */
  function boxFromLandmarks(landmarks, pad = PAD) {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of landmarks) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const bw = x1 - x0, bh = y1 - y0;
    x0 -= bw * pad; y0 -= bh * pad; x1 += bw * pad; y1 += bh * pad;
    return {
      nx0: Math.max(0, x0), ny0: Math.max(0, y0),
      nx1: Math.min(1, x1), ny1: Math.min(1, y1),
    };
  }

  /* ---------- classification ---------- */
  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = Array.from(logits, (v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  async function classify(source, sx, sy, sw, sh) {
    cropCtx.drawImage(source, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
    const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);
    const plane = SIZE * SIZE;
    const f = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      f[i] = data[p] / 255;
      f[i + plane] = data[p + 1] / 255;
      f[i + 2 * plane] = data[p + 2] / 255;
    }
    const tensor = new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]);
    const out = await session.run({ images: tensor });
    const logits = out[session.outputNames[0]].data;
    const probs = softmax(logits);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { count: best, conf: probs[best], probs };
  }

  /* ---------- drawing ---------- */
  function drawOverlay(box, result) {
    const w = overlay.clientWidth, h = overlay.clientHeight;
    overlay.width = w; overlay.height = h;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!box) return;

    // video is CSS-mirrored, so mirror the x axis when drawing
    const mirror = mode === "camera";
    const x = (mirror ? 1 - box.nx1 : box.nx0) * w;
    const y = box.ny0 * h;
    const bw = (box.nx1 - box.nx0) * w;
    const bh = (box.ny1 - box.ny0) * h;

    ctx.strokeStyle = "#4ade80";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, bw, bh);

    const label = result ? `${result.count} fingers · ${(result.conf * 100).toFixed(0)}%` : "hand";
    ctx.font = "600 15px -apple-system, Segoe UI, Roboto, sans-serif";
    const tw = ctx.measureText(label).width + 16;
    const ly = y > 28 ? y - 28 : y + 6;
    ctx.fillStyle = "#4ade80";
    ctx.fillRect(x, ly, tw, 24);
    ctx.fillStyle = "#06210f";
    ctx.fillText(label, x + 8, ly + 17);
  }

  function showResult(result) {
    countValue.textContent = result.count;
    countLabel.textContent = result.count === 1 ? "finger" : "fingers";
    const pct = Math.round(result.conf * 100);
    confFill.style.width = `${pct}%`;
    const second = result.probs
      .map((p, i) => ({ p, i }))
      .filter((o) => o.i !== result.count)
      .sort((a, b) => b.p - a.p)[0];
    confText.textContent =
      pct >= 80
        ? `${pct}% confident`
        : `${pct}% confident — could be ${second.i} (${Math.round(second.p * 100)}%)`;
  }

  /* ---------- camera loop ---------- */
  async function loop() {
    if (!running || mode !== "camera") return;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const det = landmarker.detectForVideo(video, performance.now());
      if (det.landmarks && det.landmarks.length > 0) {
        const box = boxFromLandmarks(det.landmarks[0]);
        const vw = video.videoWidth, vh = video.videoHeight;
        const sx = box.nx0 * vw, sy = box.ny0 * vh;
        const sw = (box.nx1 - box.nx0) * vw, sh = (box.ny1 - box.ny0) * vh;
        const result = await classify(video, sx, sy, sw, sh);
        drawOverlay(box, result);
        showResult(result);
        frames++;
        const now = performance.now();
        if (now - fpsStart > 800) {
          fpsEl.textContent = `${(frames / ((now - fpsStart) / 1000)).toFixed(1)} fps · on-device`;
          frames = 0; fpsStart = now;
        }
      } else {
        drawOverlay(null, null);
        countValue.textContent = "–";
        countLabel.textContent = "fingers";
        confFill.style.width = "0%";
        confText.textContent = "no hand in frame";
      }
    }
    requestAnimationFrame(loop);
  }

  /* ---------- actions ---------- */
  async function startCamera() {
    try {
      await init();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      mode = "camera";
      photo.hidden = true;
      video.style.display = "block";
      placeholder.style.display = "none";
      if (landmarker.setOptions) await landmarker.setOptions({ runningMode: "VIDEO" });
      running = true;
      fpsStart = performance.now(); frames = 0;
      requestAnimationFrame(loop);
    } catch (err) {
      setStatus(`camera error: ${err.message}. Check browser permissions, or upload a photo instead.`);
    }
  }

  async function handlePhoto(file) {
    try {
      await init();
      const url = URL.createObjectURL(file);
      photo.src = url;
      await photo.decode();
      mode = "photo";
      running = false;
      video.style.display = "none";
      photo.hidden = false;
      placeholder.style.display = "none";
      if (landmarker.setOptions) await landmarker.setOptions({ runningMode: "IMAGE" });
      const det = landmarker.detect(photo);
      if (!det.landmarks || !det.landmarks.length) {
        drawOverlay(null, null);
        confText.textContent = "no hand found in that image";
        setStatus("try a photo where the hand is large and well lit");
        return;
      }
      const box = boxFromLandmarks(det.landmarks[0]);
      const iw = photo.naturalWidth, ih = photo.naturalHeight;
      const result = await classify(
        photo,
        box.nx0 * iw, box.ny0 * ih,
        (box.nx1 - box.nx0) * iw, (box.ny1 - box.ny0) * ih
      );
      drawOverlay(box, result);
      showResult(result);
      setStatus(`analysed ${file.name}`);
    } catch (err) {
      setStatus(`error: ${err.message}`);
    }
  }

  $("btn-cam").addEventListener("click", startCamera);
  $("file").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) handlePhoto(e.target.files[0]);
  });

  /* ---------- headless self-test: index.html?autotest=1 ---------- */
  if (new URLSearchParams(location.search).has("autotest")) {
    (async () => {
      try {
        await init();
        const c = document.createElement("canvas");
        c.width = c.height = SIZE;
        const cx = c.getContext("2d");
        cx.fillStyle = "#9a7b62"; cx.fillRect(0, 0, SIZE, SIZE);
        cx.fillStyle = "#c49a7a";
        for (let i = 0; i < 5; i++) cx.fillRect(40 + i * 30, 40, 18, 110);
        const r = await classify(c, 0, 0, SIZE, SIZE);
        document.body.dataset.autotest = JSON.stringify(r);
        setStatus(`autotest ok · predicted ${r.count} at ${(r.conf * 100).toFixed(1)}%`);
      } catch (e) {
        document.body.dataset.autotest = "failed";
        setStatus(`autotest failed: ${e.message}`);
      }
    })();
  }
})();
