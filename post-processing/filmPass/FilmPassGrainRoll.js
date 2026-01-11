// Author: Riccardo Petrini
import { ShaderMaterial, UniformsUtils } from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const FilmGrainRollShader = {
  uniforms: {
    tDiffuse: { value: null },
    time: { value: 0 },
    grain: { value: 0.65 },
    roll: { value: 0.15 },
    vignette: { value: 0.35 },
    grayscale: { value: false }
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float time;
    uniform float grain;
    uniform float roll;
    uniform float vignette;
    uniform bool grayscale;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
    void main(){
      vec2 uv = vUv;
      uv.y = fract(uv.y + roll * time * 0.02);
      vec3 color = texture2D(tDiffuse, uv).rgb;
      float g = hash(uv * vec2(time * 60.0, time * 40.0));
      color += (g - 0.5) * grain;
      float v = smoothstep(0.0, vignette, length(vUv - 0.5));
      color *= mix(1.0, 1.0 - v, 0.65);
      if (grayscale) {
        float l = dot(color, vec3(0.299,0.587,0.114));
        color = vec3(l);
      }
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

export class FilmPassGrainRoll extends Pass {
  constructor({ grain = 0.65, roll = 0.15, vignette = 0.35, grayscale = false } = {}) {
    super();
    this.uniforms = UniformsUtils.clone(FilmGrainRollShader.uniforms);
    this.uniforms.grain.value = grain;
    this.uniforms.roll.value = roll;
    this.uniforms.vignette.value = vignette;
    this.uniforms.grayscale.value = grayscale;

    this.material = new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: FilmGrainRollShader.vertexShader,
      fragmentShader: FilmGrainRollShader.fragmentShader
    });

    this._fsQuad = new FullScreenQuad(this.material);
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
