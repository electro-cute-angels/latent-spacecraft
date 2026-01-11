// Author: Riccardo Petrini
// Beta(alpha, beta) using two gamma draws
function sampleGamma(shape, scale = 1) {
  // Marsaglia and Tsang's method for k > 0
  const d = shape < 1 ? shape + (1 / 3) : shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      const u1 = Math.random();
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      x = r;
      v = Math.pow(1 + c * x, 3);
    } while (v <= 0);
    const u = Math.random();
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return d * v * scale;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
  }
}

export function sampleBeta(count = 100, alpha = 2, beta = 5) {
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    const x = sampleGamma(alpha, 1);
    const y = sampleGamma(beta, 1);
    out[i] = x / (x + y);
  }
  return out;
}
