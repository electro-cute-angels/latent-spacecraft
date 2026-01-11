// Author: Riccardo Petrini
const gridEl = document.getElementById('grid');
const tooltipEl = document.getElementById('tooltip');
const previewEl = document.getElementById('preview');
const selectedEl = document.getElementById('selected');
const stepInput = document.getElementById('step');

const SIZE = 100;
const DIM = 10;
const vector = new Array(SIZE).fill(0);
let isMouseDown = false;
let lastPaintIndex = -1;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function paint(idx, delta) {
  vector[idx] = clamp(vector[idx] + delta, -1, 1);
  renderCell(idx);
  renderPreview();
}

function renderCell(idx) {
  const cell = gridEl.children[idx];
  const v = vector[idx];
  cell.textContent = v.toFixed(2);
  cell.classList.toggle('positive', v > 0.05);
  cell.classList.toggle('negative', v < -0.05);
}

function renderPreview() {
  const slice = vector.slice(0, 16).map(v => v.toFixed(2));
  previewEl.textContent = `[${slice.join(', ')}] ...`; // trimmed for brevity
}

function initGrid() {
  for (let r = 0; r < DIM; r++) {
    for (let c = 0; c < DIM; c++) {
      const idx = r * DIM + c;
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.idx = idx;
      cell.textContent = '0.00';

      cell.addEventListener('mouseenter', (e) => {
        const step = parseFloat(stepInput.value) || 0.1;
        tooltipEl.style.opacity = 1;
        selectedEl.textContent = `Index ${idx}, value ${vector[idx].toFixed(3)}`;
        if (isMouseDown && lastPaintIndex !== idx) {
          paint(idx, e.shiftKey ? -step : step);
          lastPaintIndex = idx;
        }
      });
      cell.addEventListener('mousemove', (e) => {
        tooltipEl.style.left = `${e.clientX + 12}px`;
        tooltipEl.style.top = `${e.clientY + 12}px`;
        tooltipEl.textContent = `idx ${idx} → ${vector[idx].toFixed(3)}`;
      });
      cell.addEventListener('mouseleave', () => {
        tooltipEl.style.opacity = 0;
      });
      cell.addEventListener('mousedown', (e) => {
        const step = parseFloat(stepInput.value) || 0.1;
        isMouseDown = true;
        paint(idx, e.shiftKey ? -step : step);
        lastPaintIndex = idx;
      });
      cell.addEventListener('mouseup', () => {
        isMouseDown = false;
        lastPaintIndex = -1;
      });
      cell.addEventListener('dblclick', () => {
        vector[idx] = 0;
        renderCell(idx);
        renderPreview();
      });

      gridEl.appendChild(cell);
    }
  }
  renderPreview();
}

function randomize() {
  for (let i = 0; i < SIZE; i++) {
    vector[i] = parseFloat((Math.random() * 2 - 1).toFixed(3));
    renderCell(i);
  }
  renderPreview();
}

function zero() {
  for (let i = 0; i < SIZE; i++) {
    vector[i] = 0;
    renderCell(i);
  }
  renderPreview();
}

window.addEventListener('mouseup', () => { isMouseDown = false; lastPaintIndex = -1; });

document.getElementById('randomize').addEventListener('click', randomize);
document.getElementById('zero').addEventListener('click', zero);

initGrid();
randomize();
