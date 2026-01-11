// Author: Riccardo Petrini
// Exponential with rate lambda
export function sampleExponential(count = 100, lambda = 1.0) {
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const u = Math.max(Number.EPSILON, Math.random());
    out[i] = -Math.log(u) / lambda;
  }
  return out;
}
