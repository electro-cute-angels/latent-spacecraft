// Author: Riccardo Petrini
import {
  MeshBasicNodeMaterial,
  uv,
  time,
  vec2,
  vec3,
  vec4,
  floor,
  fract,
  dot,
  max,
  sin
} from 'three/tsl';

const K1 = 0.366025403;
const K2 = 0.211324865;

const hash = (p) => fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);

const simplex2 = (p) => {
  const i = floor(p + dot(p, vec2(K1)));
  const a = p - i + dot(i, vec2(K2));
  const m = a.x > a.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  const b = a - m + vec2(K2);
  const c = a - vec2(1.0 - 2.0 * K2);

  const ha = hash(i);
  const hb = hash(i + m);
  const hc = hash(i + vec2(1.0, 1.0));

  const wa = max(0.5 - dot(a, a), 0.0);
  const wb = max(0.5 - dot(b, b), 0.0);
  const wc = max(0.5 - dot(c, c), 0.0);

  const va = ha * wa * wa * wa * wa;
  const vb = hb * wb * wb * wb * wb;
  const vc = hc * wc * wc * wc * wc;
  return (va + vb + vc) * 2.0 - 1.0;
};

export function makeSimplexNoiseMaterial() {
  const st = uv().mul(5.0).add(time().mul(0.1));
  const n = simplex2(st);
  const col = vec3(0.5 + 0.5 * n);
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
