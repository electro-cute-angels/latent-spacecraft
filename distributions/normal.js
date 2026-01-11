// Author: Riccardo Petrini
// Standard normal via Box-Muller transform
export function sampleNormal(count = 100, mean = 0, std = 1) {
  const out = new Array(count);
  for (let i = 0; i < count; i += 2) {
    const u1 = Math.max(Number.EPSILON, Math.random());
    const u2 = Math.random();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    const z0 = r * Math.cos(theta);
    const z1 = r * Math.sin(theta);
    out[i] = mean + std * z0;
    if (i + 1 < count) out[i + 1] = mean + std * z1;
  }
  return out;
}
