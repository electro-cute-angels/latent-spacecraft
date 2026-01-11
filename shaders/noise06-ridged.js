// Author: Riccardo Petrini
import {
  MeshBasicNodeMaterial,
  uv,
  time,
  vec2,
  vec3,
  vec4,
  abs,
  sin,
  fract,
  floor,
  mix,
  dot
} from 'three/tsl';

const hash2 = (p) => fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);

const valueNoise = (p) => {
  const i = floor(p);
  const f = fract(p);
  const u = f * f * (vec2(3.0).sub(f * 2.0));
  const a = hash2(i + vec2(0.0, 0.0));
  const b = hash2(i + vec2(1.0, 0.0));
  const c = hash2(i + vec2(0.0, 1.0));
  const d = hash2(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
};

const ridged = (p) => {
  let v = 0.0;
  let amp = 0.6;
  let freq = 1.0;
  for (let i = 0; i < 5; i++) {
    const n = 1.0 - abs(valueNoise(p * freq) * 2.0 - 1.0);
    v += n * amp;
    freq *= 2.0;
    amp *= 0.5;
  }
  return v;
};

export function makeRidgedNoiseMaterial() {
  const st = uv().mul(4.0).add(time().mul(0.03));
  const n = ridged(st);
  const col = vec3(n * 0.9 + 0.1);
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
