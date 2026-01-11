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
  mix,
  normalize,
  sin
} from 'three/tsl';

const hash2 = (p) => fract(sin(dot(p, vec2(41.0, 289.0))) * 43758.5);

const grad = (p) => normalize(hash2(p).mul(6.28318).sin());

const perlin = (p) => {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(vec2(3.0).sub(f.mul(2.0)));

  const g00 = grad(i + vec2(0.0, 0.0));
  const g10 = grad(i + vec2(1.0, 0.0));
  const g01 = grad(i + vec2(0.0, 1.0));
  const g11 = grad(i + vec2(1.0, 1.0));

  const d00 = dot(g00, f - vec2(0.0, 0.0));
  const d10 = dot(g10, f - vec2(1.0, 0.0));
  const d01 = dot(g01, f - vec2(0.0, 1.0));
  const d11 = dot(g11, f - vec2(1.0, 1.0));

  return mix(mix(d00, d10, u.x), mix(d01, d11, u.x), u.y);
};

export function makePerlinNoiseMaterial() {
  const st = uv().mul(3.0).add(time().mul(0.08));
  const n = perlin(st);
  const col = vec3(0.5).add(vec3(n));
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
