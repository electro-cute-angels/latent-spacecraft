// Author: Riccardo Petrini
const banksEl = document.getElementById('banks');
const previewEl = document.getElementById('preview');
const focusedEl = document.getElementById('focused');

const SIZE = 100;
const values = new Array(SIZE).fill(0);

function renderBanks() {
  for (let b = 0; b < 10; b++) {
    const bank = document.createElement('div');
    bank.className = 'bank';

    for (let i = 0; i < 10; i++) {
      const idx = b * 10 + i;
      const wrapper = document.createElement('div');
      wrapper.className = 'slider';

      const input = document.createElement('input');
      input.type = 'range';
      input.min = -1;
      input.max = 1;
      input.step = 0.01;
      input.value = values[idx];
      input.dataset.idx = idx;

      input.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        values[idx] = parseFloat(val.toFixed(2));
        focusedEl.textContent = `Index ${idx} → ${values[idx].toFixed(2)}`;
        renderPreview();
      });
      input.addEventListener('dblclick', () => {
        values[idx] = 0;
        input.value = 0;
        focusedEl.textContent = `Index ${idx} → 0`;
        renderPreview();
      });
      input.addEventListener('click', (e) => {
        if (!e.shiftKey) return;
        e.preventDefault();
        values[idx] = parseFloat((values[idx] + 0.01 * (e.altKey ? -1 : 1)).toFixed(2));
        values[idx] = Math.max(-1, Math.min(1, values[idx]));
        input.value = values[idx];
        focusedEl.textContent = `Index ${idx} → ${values[idx].toFixed(2)}`;
        renderPreview();
      });

      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = idx;

      wrapper.appendChild(input);
      wrapper.appendChild(label);
      bank.appendChild(wrapper);
    }

    banksEl.appendChild(bank);
  }
}

function renderPreview() {
  const slice = values.slice(0, 16).map(v => v.toFixed(2));
  previewEl.textContent = `[${slice.join(', ')}] ...`;
}

function randomize() {
  for (let i = 0; i < SIZE; i++) {
    values[i] = parseFloat((Math.random() * 2 - 1).toFixed(2));
  }
  syncInputs();
  renderPreview();
}

function zero() {
  values.fill(0);
  syncInputs();
  renderPreview();
}

function spread() {
  for (let i = 0; i < SIZE; i++) {
    values[i] = parseFloat((i / (SIZE - 1) * 2 - 1).toFixed(2));
  }
  syncInputs();
  renderPreview();
}

function syncInputs() {
  document.querySelectorAll('input[type="range"]').forEach((el) => {
    const idx = parseInt(el.dataset.idx, 10);
    el.value = values[idx];
  });
}

document.getElementById('randomize').addEventListener('click', randomize);
document.getElementById('zero').addEventListener('click', zero);
document.getElementById('spread').addEventListener('click', spread);

renderBanks();
randomize();
renderPreview();
