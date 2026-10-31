// ternary-forge / live demo
// All model computation happens in {-1, 0, +1}. Truly.

// ============ Model ============

class TernaryLayer {
  constructor(n_in, n_out) {
    this.n_in = n_in; this.n_out = n_out;
    this.w_shadow = new Float32Array(n_in * n_out);
    this.b = new Float32Array(n_out);
    this.vw = new Float32Array(n_in * n_out);
    this.vb = new Float32Array(n_out);
    this.w_trit = new Int8Array(n_in * n_out);
    this.alpha = 1;
    this._x = null;
    this.init();
  }
  init() {
    const s = 1 / Math.sqrt(this.n_in);
    for (let i = 0; i < this.w_shadow.length; i++) this.w_shadow[i] = (Math.random() * 2 - 1) * s;
    this.b.fill(0); this.vw.fill(0); this.vb.fill(0);
    this.quantize();
  }
  quantize() {
    const W = this.w_shadow, n = W.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.abs(W[i]);
    const delta = 0.7 * (sum / n);
    let sA = 0, cA = 0;
    for (let i = 0; i < n; i++) if (Math.abs(W[i]) > delta) { sA += Math.abs(W[i]); cA++; }
    let alpha = cA > 0 ? sA / cA : 1;
    if (alpha > 1) alpha = 1;
    this.alpha = alpha;
    const T = this.w_trit;
    for (let i = 0; i < n; i++) {
      const w = W[i];
      T[i] = w > delta ? 1 : w < -delta ? -1 : 0;
    }
  }
  forward(x) {
    this.quantize();
    const out = new Float32Array(this.n_out);
    const T = this.w_trit, alpha = this.alpha, b = this.b, n_in = this.n_in;
    for (let i = 0; i < this.n_out; i++) {
      let acc = b[i];
      const off = i * n_in;
      for (let j = 0; j < n_in; j++) {
        const t = T[off + j];
        if (t === 1) acc += alpha * x[j];
        else if (t === -1) acc -= alpha * x[j];
      }
      out[i] = acc;
    }
    this._x = x;
    return out;
  }
  backward(g, lr, wd) {
    const n_in = this.n_in, n_out = this.n_out;
    const gin = new Float32Array(n_in);
    const T = this.w_trit, alpha = this.alpha, x = this._x;
    for (let i = 0; i < n_out; i++) {
      const gi = g[i];
      if (gi === 0) continue;
      const off = i * n_in;
      for (let j = 0; j < n_in; j++) {
        gin[j] += gi * T[off + j] * alpha;
        const gd = gi * x[j];
        const vw = 0.9 * this.vw[off + j] + gd;
        this.vw[off + j] = vw;
        let ws = this.w_shadow[off + j] - lr * (vw + wd * this.w_shadow[off + j]);
        if (ws > 1) ws = 1; else if (ws < -1) ws = -1;
        this.w_shadow[off + j] = ws;
      }
      const vb = 0.9 * this.vb[i] + gi;
      this.vb[i] = vb;
      this.b[i] -= lr * vb;
    }
    return gin;
  }
  setFromPacked(trits, alpha, bias) {
    this.w_trit.set(trits);
    this.alpha = alpha;
    this.b.set(bias);
    // rebuild shadow from trits so training resumes consistently
    for (let i = 0; i < trits.length; i++) this.w_shadow[i] = trits[i] * alpha;
    this.vw.fill(0); this.vb.fill(0);
  }
  getTritSample() {
    // evenly subsample 64x64 from (n_out x n_in) matrix
    const out = new Int8Array(64 * 64);
    const n_out = this.n_out, n_in = this.n_in, T = this.w_trit;
    for (let r = 0; r < 64; r++) {
      const ri = Math.floor(r * n_out / 64);
      for (let c = 0; c < 64; c++) {
        const ci = Math.floor(c * n_in / 64);
        out[r * 64 + c] = T[ri * n_in + ci];
      }
    }
    return out;
  }
}

class TernaryMLP {
  constructor(sizes) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) this.layers.push(new TernaryLayer(sizes[i], sizes[i+1]));
  }
  forward(x) {
    let h = x;
    for (let i = 0; i < this.layers.length; i++) {
      h = this.layers[i].forward(h);
      if (i < this.layers.length - 1) {
        const r = new Float32Array(h.length);
        for (let j = 0; j < h.length; j++) r[j] = h[j] > 0 ? h[j] : 0;
        h = r;
      }
    }
    return h;
  }
  trainOne(x, y, lr) {
    const h0 = this.layers[0].forward(x);
    for (let i = 0; i < h0.length; i++) if (h0[i] < 0) h0[i] = 0;
    const logits = this.layers[1].forward(h0);
    const p = softmax(logits);
    const dl = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) dl[i] = p[i];
    dl[y] -= 1;
    const dh = this.layers[1].backward(dl, lr, 1e-3);
    for (let i = 0; i < dh.length; i++) if (h0[i] <= 0) dh[i] = 0;
    this.layers[0].backward(dh, lr, 1e-3);
    return p[y];
  }
  predict(x) {
    const logits = this.forward(x);
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return { digit: best, probs: softmax(logits) };
  }
}

// ============ Utilities ============

function softmax(arr) {
  const m = Math.max(...arr);
  const e = arr.map(x => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

function unpack4bit(packed, out) {
  for (let i = 0; i < packed.length; i++) {
    out[2*i] = (packed[i] >> 4) & 0x0F;
    out[2*i+1] = packed[i] & 0x0F;
  }
  for (let i = 0; i < out.length; i++) out[i] /= 15;
}

// ============ State ============

let model = null;
let training = null; // {train_images, train_labels, test_images, test_labels, n_train, n_test}
let pretrained = null; // snapshot of loaded weights.trit for Reset button
let pretrainedLoaded = false;

// ============ Load weights.trit ============

async function loadPretrained() {
  const resp = await fetch('weights.trit');
  if (!resp.ok) throw new Error('HTTP ' + resp.status + ' on weights.trit');
  const buf = await resp.arrayBuffer();
  const dv = new DataView(buf);
  let pos = 0;

  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== 'TRIT') throw new Error('bad magic: ' + magic);
  pos = 4;
  const version = dv.getUint8(pos); pos += 1;
  if (version !== 1) throw new Error('bad version ' + version);
  const nLayers = dv.getUint8(pos); pos += 1;

  const layers = [];
  for (let L = 0; L < nLayers; L++) {
    const nIn = dv.getUint32(pos, true); pos += 4;
    const nOut = dv.getUint32(pos, true); pos += 4;
    const alpha = dv.getFloat32(pos, true); pos += 4;
    const bias = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) { bias[i] = dv.getFloat32(pos, true); pos += 4; }
    const nTrits = dv.getUint32(pos, true); pos += 4;
    const nBytes = Math.ceil(nTrits / 5);
    const packed = new Uint8Array(buf, pos, nBytes); pos += nBytes;
    const trits = new Int8Array(nTrits);
    let k = 0;
    for (let b = 0; b < nBytes; b++) {
      let n = packed[b];
      for (let t = 0; t < 5 && k < nTrits; t++) { trits[k++] = (n % 3) - 1; n = Math.floor(n / 3); }
    }
    layers.push({ n_in: nIn, n_out: nOut, alpha, bias, trits });
  }
  return { layers, bytes: buf.byteLength };
}

// ============ Load train_data.bin ============

async function loadTrainingData() {
  const resp = await fetch('train_data.bin');
  if (!resp.ok) throw new Error('HTTP ' + resp.status + ' on train_data.bin');
  const buf = await resp.arrayBuffer();
  const dv = new DataView(buf);

  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'TRNB') throw new Error('bad train magic');
  let pos = 4;
  const version = dv.getUint8(pos); pos += 1;
  if (version !== 1) throw new Error('bad train version');
  const n_train = dv.getUint32(pos, true); pos += 4;
  const n_test = dv.getUint32(pos, true); pos += 4;
  const n_pixels = dv.getUint16(pos, true); pos += 2;
  pos += 3; // padding

  const train_labels = new Uint8Array(buf, pos, n_train); pos += n_train;
  const test_labels = new Uint8Array(buf, pos, n_test); pos += n_test;

  const bytesPerImage = Math.ceil(n_pixels / 2);
  const train_images = new Uint8Array(buf, pos, n_train * bytesPerImage); pos += n_train * bytesPerImage;
  const test_images = new Uint8Array(buf, pos, n_test * bytesPerImage);

  return {
    n_train, n_test, n_pixels,
    train_labels, test_labels,
    train_images, test_images, bytesPerImage,
  };
}

function unpackSample(data, idx) {
  const out = new Float32Array(784);
  const off = idx * data.bytesPerImage;
  const sub = new Uint8Array(data.train_images.buffer, data.train_images.byteOffset + off, data.bytesPerImage);
  unpack4bit(sub, out);
  return out;
}

function unpackTestSample(data, idx) {
  const out = new Float32Array(784);
  const off = idx * data.bytesPerImage;
  const sub = new Uint8Array(data.test_images.buffer, data.test_images.byteOffset + off, data.bytesPerImage);
  unpack4bit(sub, out);
  return out;
}

// ============ Play tab ============

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const digitEl = document.getElementById('digit');
const barsEl = document.getElementById('bars');
const metaEl = document.getElementById('meta');
const statusEl = document.getElementById('status');

let drawing = false, lastX = 0, lastY = 0, lastP = 0;

function resetCanvas() {
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff';
  ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
}

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * (canvas.width / r.width),
           y: (e.clientY - r.top) * (canvas.height / r.height) };
}

canvas.addEventListener('pointerdown', e => {
  e.preventDefault(); drawing = true;
  const p = canvasPoint(e); lastX = p.x; lastY = p.y;
  ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
  predict();
});
canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  e.preventDefault();
  const p = canvasPoint(e);
  ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke();
  lastX = p.x; lastY = p.y;
  const now = Date.now();
  if (now - lastP > 60) { lastP = now; predict(); }
});
const stop = () => { drawing = false; predict(); };
canvas.addEventListener('pointerup', stop);
canvas.addEventListener('pointerleave', stop);
canvas.addEventListener('pointercancel', stop);
document.getElementById('clear').addEventListener('click', () => {
  resetCanvas(); digitEl.textContent = 'â€”'; barsEl.innerHTML = '';
});

function getPixels() {
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  const octx = off.getContext('2d');
  octx.drawImage(canvas, 0, 0, 28, 28);
  const d = octx.getImageData(0, 0, 28, 28).data;
  const p = new Float32Array(784);
  for (let i = 0; i < 784; i++) p[i] = d[i * 4] / 255;
  return p;
}

function predict() {
  if (!model) return;
  const pixels = getPixels();
  let ink = 0;
  for (const p of pixels) ink += p;
  if (ink < 3) { digitEl.textContent = 'â€”'; barsEl.innerHTML = ''; return; }
  const { digit, probs } = model.predict(pixels);
  digitEl.textContent = digit;
  barsEl.innerHTML = '';
  for (let i = 0; i < probs.length; i++) {
    const pct = (probs[i] * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar' + (i === digit ? ' top' : '');
    row.innerHTML = '<span class="label">' + i + '</span>' +
                    '<div class="track"><div class="fill" style="width:' + pct + '%"></div></div>' +
                    '<span class="pct">' + pct + '%</span>';
    barsEl.appendChild(row);
  }
}

// ============ Train tab ============

const btnTrain = document.getElementById('btn-train');
const btnReset = document.getElementById('btn-reset');
const progEl = document.getElementById('prog');
const statEpoch = document.getElementById('stat-epoch');
const statBatch = document.getElementById('stat-batch');
const statTrain = document.getElementById('stat-train');
const statTest = document.getElementById('stat-test');
const statTime = document.getElementById('stat-time');
const statZero = document.getElementById('stat-zero');
const weightViz = document.getElementById('weight-viz');
const wvCtx = weightViz.getContext('2d');
const accViz = document.getElementById('accuracy-viz');
const avCtx = accViz.getContext('2d');

function drawWeights() {
  if (!model) return;
  const trits = model.layers[0].getTritSample();
  const img = wvCtx.createImageData(64, 64);
  for (let i = 0; i < 64 * 64; i++) {
    const t = trits[i];
    let r = 0, g = 0, b = 0;
    if (t === 1) g = 255;
    else if (t === -1) r = 255;
    else { r = 20; g = 20; b = 20; }
    img.data[i*4] = r; img.data[i*4+1] = g; img.data[i*4+2] = b; img.data[i*4+3] = 255;
  }
  wvCtx.putImageData(img, 0, 0);
}

let accHistory = [];
function drawAccuracy() {
  avCtx.fillStyle = '#000';
  avCtx.fillRect(0, 0, accViz.width, accViz.height);
  avCtx.strokeStyle = '#1f1f1f';
  avCtx.lineWidth = 1;
  for (let y = 0; y <= 100; y += 25) {
    const py = accViz.height - (y / 100) * accViz.height;
    avCtx.beginPath(); avCtx.moveTo(0, py); avCtx.lineTo(accViz.width, py); avCtx.stroke();
    avCtx.fillStyle = '#333';
    avCtx.font = '9px monospace';
    avCtx.fillText(y + '%', 2, py - 2);
  }
  if (accHistory.length < 1) return;
  avCtx.strokeStyle = '#00ff88';
  avCtx.lineWidth = 2;
  avCtx.beginPath();
  for (let i = 0; i < accHistory.length; i++) {
    const x = accHistory.length === 1 ? accViz.width / 2 : (i / (accHistory.length - 1)) * accViz.width;
    const y = accViz.height - accHistory[i] * accViz.height;
    if (i === 0) avCtx.moveTo(x, y); else avCtx.lineTo(x, y);
  }
  avCtx.stroke();
}

function countZeros() {
  const T = model.layers[0].w_trit;
  let z = 0;
  for (let i = 0; i < T.length; i++) if (T[i] === 0) z++;
  return (z / T.length * 100).toFixed(1);
}

async function runTraining() {
  if (!training || !model) return;
  btnTrain.disabled = true;
  btnReset.disabled = true;

  // reset to random trits
  for (const layer of model.layers) layer.init();

  const epochs = 5;
  const lr = 0.01;
  const startTime = performance.now();
  accHistory = [];
  drawWeights(); drawAccuracy();

  for (let ep = 0; ep < epochs; ep++) {
    // shuffle
    const order = new Uint32Array(training.n_train);
    for (let i = 0; i < training.n_train; i++) order[i] = i;
    for (let i = training.n_train - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let trainCorrect = 0;
    for (let i = 0; i < training.n_train; i++) {
      const idx = order[i];
      const x = unpackSample(training, idx);
      const y = training.train_labels[idx];
      const py = model.trainOne(x, y, lr);
      if (py > 0.5) trainCorrect++;

      if (i % 50 === 0 || i === training.n_train - 1) {
        const prog = ((ep * training.n_train + i + 1) / (epochs * training.n_train)) * 100;
        progEl.style.width = prog + '%';
        statEpoch.textContent = (ep + 1) + ' / ' + epochs;
        statBatch.textContent = i + ' / ' + training.n_train;
        statTime.textContent = ((performance.now() - startTime) / 1000).toFixed(1) + 's';
        statZero.textContent = countZeros() + '%';
        await new Promise(r => setTimeout(r, 0));
        drawWeights();
      }
    }

    // evaluate on test set
    let correct = 0;
    for (let i = 0; i < training.n_test; i++) {
      const x = unpackTestSample(training, i);
      const { digit } = model.predict(x);
      if (digit === training.test_labels[i]) correct++;
    }
    const testAcc = correct / training.n_test;
    accHistory.push(testAcc);
    statTrain.textContent = (trainCorrect / training.n_train * 100).toFixed(1) + '%';
    statTest.textContent = (testAcc * 100).toFixed(1) + '%';
    drawAccuracy();
    drawWeights();
    await new Promise(r => setTimeout(r, 10));
  }

  progEl.style.width = '100%';
  statTime.textContent = ((performance.now() - startTime) / 1000).toFixed(1) + 's';
  btnTrain.disabled = false;
  btnReset.disabled = false;

  // switch back to play tab so user can try the new weights
  activateTab('play');
}

async function resetToPretrained() {
  if (!pretrained || !model) return;
  btnTrain.disabled = true;
  btnReset.disabled = true;
  for (let i = 0; i < pretrained.layers.length; i++) {
    const L = pretrained.layers[i];
    model.layers[i].setFromPacked(L.trits, L.alpha, L.bias);
  }
  accHistory = [];
  statEpoch.textContent = 'â€”'; statBatch.textContent = 'â€”';
  statTrain.textContent = 'â€”'; statTest.textContent = 'â€”';
  statTime.textContent = 'â€”';
  statZero.textContent = countZeros() + '%';
  progEl.style.width = '0%';
  drawWeights(); drawAccuracy();
  btnTrain.disabled = false;
  btnReset.disabled = true;
}

// ============ Tabs ============

document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => activateTab(t.dataset.tab));
});

function activateTab(name) {
  if (name === 'train' && model && training && !btnTrain.disabled && accHistory.length === 0) {
    setTimeout(() => runTraining(), 150);
  }
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.section').forEach(s => s.classList.toggle('active', s.id === 'sec-' + name));
}

// ============ Init ============

async function init() {
  try {
    const weights = await loadPretrained();
    model = new TernaryMLP([784, 256, 10]);
    for (let i = 0; i < weights.layers.length; i++) {
      const L = weights.layers[i];
      model.layers[i].setFromPacked(L.trits, L.alpha, L.bias);
    }
    pretrained = weights;

    try {
      training = await loadTrainingData();
      btnTrain.disabled = false;
      statusEl.textContent =
        'Pre-trained model loaded (' + (weights.bytes / 1024).toFixed(1) + ' KiB weights, ' +
        (training.n_train + training.n_test) + ' training samples ready)';
    } catch (e) {
      training = null;
      btnTrain.disabled = true;
      btnTrain.textContent = 'train_data.bin missing';
      statusEl.textContent = 'Pre-trained model loaded (training data unavailable: ' + e.message + ')';
    }
    statusEl.className = 'ok';

    statZero.textContent = countZeros() + '%';
    drawWeights();
    drawAccuracy();
  } catch (e) {
    statusEl.textContent = 'Failed to load: ' + e.message + ' (serve over HTTP, not file://)';
    statusEl.className = 'err';
    console.error(e);
  }
}

btnTrain.addEventListener('click', runTraining);
btnReset.addEventListener('click', resetToPretrained);

resetCanvas();
init();