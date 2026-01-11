// Author: Riccardo Petrini
import {
  MeshBasicNodeMaterial,
  uv,
  time,
  vec2,
  vec3,
  vec4,
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

export function makeValueNoiseMaterial() {
  const st = uv().mul(4.0).add(time().mul(0.05));
  const n = valueNoise(st);
  const col = vec3(n).mul(0.8).add(vec3(0.1, 0.2, 0.3));
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
