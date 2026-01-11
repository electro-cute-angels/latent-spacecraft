// Author: Riccardo Petrini
const canvas = document.getElementById('wheel');
const ctx = canvas.getContext('2d');
const hoverEl = document.getElementById('hover');
const previewEl = document.getElementById('preview');

const SIZE = 100;
const values = new Array(SIZE).fill(0);
let hoverIdx = -1;
let dragging = false;

function polarToCartesian(cx, cy, r, angle) {
  return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
}

function draw() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();

  const cx = w / 2;
  const cy = h / 2;
  const inner = 80;
  const outer = Math.min(w, h) * 0.45;
  const step = (Math.PI * 2) / SIZE;

  // base circle
  ctx.strokeStyle = '#112233';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, Math.PI * 2);
  ctx.stroke();

  for (let i = 0; i < SIZE; i++) {
    const angle = -Math.PI / 2 + i * step;
    const v = values[i];
    const radius = inner + ((v + 1) / 2) * (outer - inner);
    const [x0, y0] = polarToCartesian(cx, cy, inner, angle);
    const [x1, y1] = polarToCartesian(cx, cy, radius, angle);

    const isHover = i === hoverIdx;
    ctx.strokeStyle = isHover ? '#7cffb2' : '#4de2ff';
    ctx.lineWidth = isHover ? 3 : 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    // tip dot
    ctx.fillStyle = isHover ? '#7cffb2' : '#4de2ff';
    ctx.beginPath();
    ctx.arc(x1, y1, isHover ? 4 : 3, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function setHover(idx) {
  hoverIdx = idx;
  if (idx === -1) {
    hoverEl.textContent = 'None';
  } else {
    hoverEl.textContent = `Index ${idx} → ${values[idx].toFixed(3)}`;
  }
  draw();
}

function updateValueFromPointer(ev) {
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const dx = x - cx;
  const dy = y - cy;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx);

  const step = (Math.PI * 2) / SIZE;
  let idx = Math.round(((angle + Math.PI / 2 + Math.PI * 2) % (Math.PI * 2)) / step);
  if (idx >= SIZE) idx = 0;
  setHover(idx);

  const inner = 80;
  const outer = Math.min(canvas.width, canvas.height) * 0.45;
  const norm = (dist - inner) / (outer - inner);
  const v = Math.max(-1, Math.min(1, norm * 2 - 1));
  if (dragging) {
    values[idx] = parseFloat(v.toFixed(3));
    renderPreview();
  }
}

function renderPreview() {
  const slice = values.slice(0, 16).map(v => v.toFixed(2));
  previewEl.textContent = `[${slice.join(', ')}] ...`;
}

function randomize() {
  for (let i = 0; i < SIZE; i++) {
    values[i] = parseFloat((Math.random() * 2 - 1).toFixed(3));
  }
  renderPreview();
  draw();
}

function zero() {
  values.fill(0);
  renderPreview();
  draw();
}

canvas.addEventListener('mousedown', (e) => { dragging = true; updateValueFromPointer(e); });
canvas.addEventListener('mousemove', (e) => updateValueFromPointer(e));
window.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('mouseleave', () => setHover(-1));

document.getElementById('randomize').addEventListener('click', randomize);
document.getElementById('reset').addEventListener('click', zero);

randomize();
setHover(-1);
