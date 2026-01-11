// Author: Riccardo Petrini
import { ShaderMaterial, UniformsUtils, Vector2 } from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const FilmChromaticShader = {
  uniforms: {
    tDiffuse: { value: null },
    time: { value: 0 },
    grain: { value: 0.4 },
    shift: { value: 1.0 },
    scanlines: { value: 0.2 },
    resolution: { value: new Vector2(1024, 1024) }
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float time;
    uniform float grain;
    uniform float shift;
    uniform float scanlines;
    uniform vec2 resolution;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }
    void main(){
      vec2 texel = 1.0 / resolution;
      vec2 offset = vec2(shift) * texel;
      vec3 col;
      col.r = texture2D(tDiffuse, vUv + offset * 0.6).r;
      col.g = texture2D(tDiffuse, vUv).g;
      col.b = texture2D(tDiffuse, vUv - offset * 0.6).b;
      float g = (hash(vUv * vec2(time*23.0, time*47.0)) - 0.5) * grain;
      col += g;
      float s = sin(vUv.y * resolution.y * 3.14159) * scanlines;
      col *= 1.0 - s * 0.2;
      gl_FragColor = vec4(col, 1.0);
    }
  `
};

export class FilmPassChromatic extends Pass {
  constructor({ grain = 0.4, shift = 1.0, scanlines = 0.2 } = {}) {
    super();
    this.uniforms = UniformsUtils.clone(FilmChromaticShader.uniforms);
    this.uniforms.grain.value = grain;
    this.uniforms.shift.value = shift;
    this.uniforms.scanlines.value = scanlines;

    this.material = new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: FilmChromaticShader.vertexShader,
      fragmentShader: FilmChromaticShader.fragmentShader
    });
    this._fsQuad = new FullScreenQuad(this.material);
  }

  setSize(w, h) {
    this.uniforms.resolution.value.set(w, h);
  }

  render(renderer, writeBuffer, readBuffer, deltaTime) {
    this.uniforms.tDiffuse.value = readBuffer.texture;
    this.uniforms.time.value += deltaTime;

    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
      this._fsQuad.render(renderer);
    } else {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
      this._fsQuad.render(renderer);
    }
  }

  dispose() {
    this.material.dispose();
    this._fsQuad.dispose();
  }
}
