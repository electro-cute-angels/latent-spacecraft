// Author: Riccardo Petrini
import { Vector2 } from 'three';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

const shader = {
  uniforms: {
    tDiffuse: { value: null },
    resolution: { value: new Vector2(1024, 1024) },
    time: { value: 0 },
    pixelSize: { value: 6.0 },
    speed: { value: 0.6 },
    width: { value: 0.18 },
    tail: { value: 0.35 }
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2 resolution;
    uniform float time;
    uniform float pixelSize;
    uniform float speed;
    uniform float width;
    uniform float tail;
    varying vec2 vUv;
    void main(){
      vec2 px = floor(vUv * resolution / pixelSize) * pixelSize / resolution;
      vec4 texel = texture2D(tDiffuse, px);
      float pos = fract(time * speed);
      float coord = fract((vUv.x + vUv.y) * 0.5);
      float d = coord - pos;
      if (d < 0.0) d += 1.0;
      float vis = 0.0;
      if (d < width) vis = 1.0;
      else if (d < width + tail) vis = 1.0 - (d - width) / tail;
      if (vis <= 0.0) discard;
      gl_FragColor = vec4(texel.rgb * vis, 1.0);
    }
  `
};

export class ScanPixelPassDiagonal extends ShaderPass {
  constructor({ pixelSize = 6.0, speed = 0.6, width = 0.18, tail = 0.35 } = {}) {
    super(shader);
    this.uniforms.pixelSize.value = pixelSize;
    this.uniforms.speed.value = speed;
    this.uniforms.width.value = width;
    this.uniforms.tail.value = tail;
    this._start = performance.now();
  }

  render(renderer, writeBuffer, readBuffer, deltaTime) {
    this.uniforms.time.value = (performance.now() - this._start) * 0.001;
    super.render(renderer, writeBuffer, readBuffer, deltaTime);
  }

  setSize(w, h) {
    this.uniforms.resolution.value.set(w, h);
  }
}
