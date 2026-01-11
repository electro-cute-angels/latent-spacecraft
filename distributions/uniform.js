// Author: Riccardo Petrini
// Uniform distribution on [min, max]
export function sampleUniform(count = 100, min = -1, max = 1) {
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = min + Math.random() * (max - min);
  }
  return out;
}
