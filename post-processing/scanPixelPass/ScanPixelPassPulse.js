// Author: Riccardo Petrini
import { Vector2 } from 'three';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

const shader = {
  uniforms: {
    tDiffuse: { value: null },
    resolution: { value: new Vector2(1024, 1024) },
    time: { value: 0 },
    pixelSize: { value: 10.0 },
    speed: { value: 0.8 },
    band: { value: 0.25 },
    pulse: { value: 1.6 }
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2 resolution;
    uniform float time;
    uniform float pixelSize;
    uniform float speed;
    uniform float band;
    uniform float pulse;
    varying vec2 vUv;
    void main(){
      vec2 px = floor(vUv * resolution / pixelSize) * pixelSize / resolution;
      vec4 texel = texture2D(tDiffuse, px);
      float pos = fract(time * speed);
      float dist = abs(vUv.y - pos);
      float glow = smoothstep(band, 0.0, dist) * (0.6 + 0.4*sin(time * pulse));
      if (glow <= 0.01) discard;
      gl_FragColor = vec4(texel.rgb * glow, 1.0);
    }
  `
};

export class ScanPixelPassPulse extends ShaderPass {
  constructor({ pixelSize = 10.0, speed = 0.8, band = 0.25, pulse = 1.6 } = {}) {
    super(shader);
    this.uniforms.pixelSize.value = pixelSize;
    this.uniforms.speed.value = speed;
    this.uniforms.band.value = band;
    this.uniforms.pulse.value = pulse;
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
