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
  sin,
  dot,
  mix,
  length,
  normalize
} from 'three/tsl';

const hash2 = (p) => fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);

const worley = (p) => {
  const i = floor(p);
  let d = 1.0;
  for (let y = -1; y <= 1; y++) {
    for (let x = -1; x <= 1; x++) {
      const cell = i + vec2(x, y);
      const rnd = hash2(cell);
      const feature = cell + rnd;
      d = min(d, length(feature - p));
    }
  }
  return d;
};

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

export function makeCellularBlendMaterial() {
  const st = uv().mul(5.5).add(time().mul(0.06));
  const w = worley(st);
  const p = perlin(st.mul(0.6));
  const n = mix(1.0 - w * 0.8, p * 0.5 + 0.5, 0.5);
  const col = vec3(n * 0.9 + 0.05);
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
