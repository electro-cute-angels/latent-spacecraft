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
  length,
  sin,
  dot,
  min
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

export function makeWorleyNoiseMaterial() {
  const st = uv().mul(6.0).add(time().mul(0.05));
  const d = worley(st);
  const col = vec3(1.0 - d * 0.8);
  return new MeshBasicNodeMaterial({ colorNode: vec4(col, 1.0) });
}
