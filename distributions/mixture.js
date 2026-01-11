// Author: Riccardo Petrini
// Simple Gaussian mixture: two normals blended by weight
import { sampleNormal } from './normal.js';

export function sampleMixture(count = 100, weight = 0.5, a = { mean: -1, std: 0.5 }, b = { mean: 1, std: 0.5 }) {
  const out = new Array(count);
  const aCount = Math.round(count * weight);
  const bCount = count - aCount;
  const aSamples = sampleNormal(aCount, a.mean, a.std);
  const bSamples = sampleNormal(bCount, b.mean, b.std);
  for (let i = 0; i < aCount; i++) out[i] = aSamples[i];
  for (let i = 0; i < bCount; i++) out[aCount + i] = bSamples[i];
  return out;
}
