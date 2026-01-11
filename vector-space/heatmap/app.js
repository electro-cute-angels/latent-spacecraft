// Author: Riccardo Petrini
const canvas = document.getElementById('heatmap');
const ctx = canvas.getContext('2d');
const radiusInput = document.getElementById('radius');
const statusEl = document.getElementById('status');
const previewEl = document.getElementById('preview');

const SIZE = 100;
const DIM = 10;
const values = new Array(SIZE).fill(0);
let frozen = false;
let pointer = { x: canvas.width / 2, y: canvas.height / 2 };

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function colorFor(v) {
  const t = (clamp(v, -1, 1) + 1) / 2; // 0..1
  const hue = 190 - 150 * t; // teal to amber
  const light = 20 + 45 * t;
  return `hsl(${hue}, 80%, ${light}%)`;
}

function updateField(px = pointer.x, py = pointer.y) {
  if (frozen) return;
  pointer = { x: px, y: py };
  const cellW = canvas.width / DIM;
  const cellH = canvas.height / DIM;
  const radiusPx = parseFloat(radiusInput.value) || 140;

  for (let r = 0; r < DIM; r++) {
    for (let c = 0; c < DIM; c++) {
      const idx = r * DIM + c;
      const cx = c * cellW + cellW / 2;
      const cy = r * cellH + cellH / 2;
      const dx = px - cx;
      const dy = py - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const strength = clamp(1 - dist / radiusPx, 0, 1);
      const v = Math.pow(strength, 1.5) * 2 - 1; // -1..1, peaked under cursor
      values[idx] = parseFloat(v.toFixed(3));
    }
  }
  renderPreview();
  draw();
}

function draw() {
  const cellW = canvas.width / DIM;
  const cellH = canvas.height / DIM;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < DIM; r++) {
    for (let c = 0; c < DIM; c++) {
      const idx = r * DIM + c;
      ctx.fillStyle = colorFor(values[idx]);
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let i = 1; i < DIM; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cellW, 0);
    ctx.lineTo(i * cellW, canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * cellH);
    ctx.lineTo(canvas.width, i * cellH);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(107,255,181,0.9)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(pointer.x, pointer.y, parseFloat(radiusInput.value), 0, Math.PI * 2);
  ctx.stroke();
}

function renderPreview() {
  const slice = values.slice(0, 16).map(v => v.toFixed(2));
  previewEl.textContent = `[${slice.join(', ')}] ...`;
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  statusEl.textContent = `Cursor (${x.toFixed(0)}, ${y.toFixed(0)})`;
  updateField(x, y);
});

canvas.addEventListener('mouseleave', () => {
  statusEl.textContent = frozen ? 'Frozen' : 'Move cursor over the field';
});

radiusInput.addEventListener('input', () => {
  statusEl.textContent = `Radius ${radiusInput.value}px`;
  updateField(pointer.x, pointer.y);
});

document.getElementById('freeze').addEventListener('click', () => {
  frozen = !frozen;
  statusEl.textContent = frozen ? 'Frozen' : 'Live';
});

document.getElementById('reset').addEventListener('click', () => {
  values.fill(0);
  renderPreview();
  draw();
  statusEl.textContent = 'Reset to zeros';
});

document.getElementById('center').addEventListener('click', () => {
  const x = canvas.width / 2;
  const y = canvas.height / 2;
  frozen = false;
  updateField(x, y);
  statusEl.textContent = 'Centered';
});

window.addEventListener('resize', () => {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.width * 0.666 * devicePixelRatio;
  updateField(pointer.x, pointer.y);
});

// Initial layout
(() => {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.width * 0.666 * devicePixelRatio;
  updateField(pointer.x, pointer.y);
  renderPreview();
})();
