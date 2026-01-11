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

const fbm = (p) => {
  let v = 0.0;
  let amp = 0.5;
  let freq = 1.2;
  for (let i = 0; i < 4; i++) {
    v += valueNoise(p * freq) * amp;
    freq *= 2.1;
    amp *= 0.55;
  }
  return v;
};

export function makeFlowNoiseMaterial() {
  const t = time().mul(0.12);
  const flow = (coords) => {
    const angle = fbm(coords.add(vec2(0.0, t))) * 6.28318;
    const dir = vec2(angle.cos(), angle.sin());
    return coords.add(dir.mul(0.2));
  };

  let st = uv().mul(3.0);
  st = flow(st);
  st = flow(st.add(vec2(0.3)));
  const n = fbm(st.add(vec2(t)));
  const col = vec3(0.2, 0.4, 0.6).add(vec3(n * 0.8));
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
