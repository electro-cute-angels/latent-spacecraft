// Author: Riccardo Petrini
// Lognormal: exp of a normal sample
export function sampleLogNormal(count = 100, mu = 0, sigma = 0.5) {
  const out = new Array(count);
  for (let i = 0; i < count; i += 2) {
    const u1 = Math.max(Number.EPSILON, Math.random());
    const u2 = Math.random();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    const z0 = r * Math.cos(theta) * sigma + mu;
    const z1 = r * Math.sin(theta) * sigma + mu;
    out[i] = Math.exp(z0);
    if (i + 1 < count) out[i + 1] = Math.exp(z1);
  }
  return out;
}
