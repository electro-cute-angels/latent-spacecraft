// Author: Riccardo Petrini
// Dual-threshold bloom with small separable blur
import {
  AdditiveBlending,
  HalfFloatType,
  ShaderMaterial,
  UniformsUtils,
  WebGLRenderTarget,
  Vector2
} from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const ThresholdShader = {
  uniforms: {
    tDiffuse: { value: null },
    threshold: { value: 1.0 },
    softKnee: { value: 0.5 }
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float threshold;
    uniform float softKnee;
    varying vec2 vUv;
    void main(){
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float l = max(max(c.r, c.g), c.b);
      float knee = threshold * softKnee + 1e-4;
      float soft = smoothstep(threshold - knee, threshold + knee, l);
      gl_FragColor = vec4(c * soft, 1.0);
    }
  `
};

const BlurShader = {
  uniforms: {
    tDiffuse: { value: null },
    direction: { value: new Vector2(1.0, 0.0) },
    resolution: { value: new Vector2(1024, 1024) }
  },
  vertexShader: ThresholdShader.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2 direction;
    uniform vec2 resolution;
    varying vec2 vUv;
    void main(){
      vec2 texel = direction / resolution;
      vec3 col = vec3(0.0);
      col += texture2D(tDiffuse, vUv - 2.0*texel).rgb * 0.12;
      col += texture2D(tDiffuse, vUv - 1.0*texel).rgb * 0.23;
      col += texture2D(tDiffuse, vUv).rgb * 0.30;
      col += texture2D(tDiffuse, vUv + 1.0*texel).rgb * 0.23;
      col += texture2D(tDiffuse, vUv + 2.0*texel).rgb * 0.12;
      gl_FragColor = vec4(col, 1.0);
    }
  `
};

const CombineShader = {
  uniforms: {
    tScene: { value: null },
    tBloom: { value: null },
    intensity: { value: 1.0 }
  },
  vertexShader: ThresholdShader.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tScene;
    uniform sampler2D tBloom;
    uniform float intensity;
    varying vec2 vUv;
    void main(){
      vec3 scene = texture2D(tScene, vUv).rgb;
      vec3 bloom = texture2D(tBloom, vUv).rgb * intensity;
      gl_FragColor = vec4(scene + bloom, 1.0);
    }
  `
};

export class BloomPassDual extends Pass {
  constructor(threshold = 1.0, softKnee = 0.5, intensity = 1.2) {
    super();

    this.thresholdMaterial = new ShaderMaterial({
      uniforms: UniformsUtils.clone(ThresholdShader.uniforms),
      vertexShader: ThresholdShader.vertexShader,
      fragmentShader: ThresholdShader.fragmentShader
    });
    this.thresholdMaterial.uniforms.threshold.value = threshold;
    this.thresholdMaterial.uniforms.softKnee.value = softKnee;

    this.blurMaterial = new ShaderMaterial({
      uniforms: UniformsUtils.clone(BlurShader.uniforms),
      vertexShader: BlurShader.vertexShader,
      fragmentShader: BlurShader.fragmentShader
    });

    this.combineMaterial = new ShaderMaterial({
      uniforms: UniformsUtils.clone(CombineShader.uniforms),
      vertexShader: CombineShader.vertexShader,
      fragmentShader: CombineShader.fragmentShader,
      blending: AdditiveBlending,
      transparent: true
    });
    this.combineMaterial.uniforms.intensity.value = intensity;

    this._rtBright = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
    this._rtBlurX = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
    this._rtBlurY = new WebGLRenderTarget(1, 1, { type: HalfFloatType });

    this._fsQuad = new FullScreenQuad(null);
  }

  render(renderer, writeBuffer, readBuffer) {
    // threshold
    this._fsQuad.material = this.thresholdMaterial;
    this.thresholdMaterial.uniforms.tDiffuse.value = readBuffer.texture;
    renderer.setRenderTarget(this._rtBright);
    renderer.clear();
    this._fsQuad.render(renderer);

    // blur X
    this._fsQuad.material = this.blurMaterial;
    this.blurMaterial.uniforms.tDiffuse.value = this._rtBright.texture;
    this.blurMaterial.uniforms.direction.value.set(1.0, 0.0);
    renderer.setRenderTarget(this._rtBlurX);
    renderer.clear();
    this._fsQuad.render(renderer);

    // blur Y
    this.blurMaterial.uniforms.tDiffuse.value = this._rtBlurX.texture;
    this.blurMaterial.uniforms.direction.value.set(0.0, 1.0);
    renderer.setRenderTarget(this._rtBlurY);
    renderer.clear();
    this._fsQuad.render(renderer);

    // combine
    this._fsQuad.material = this.combineMaterial;
    this.combineMaterial.uniforms.tScene.value = readBuffer.texture;
    this.combineMaterial.uniforms.tBloom.value = this._rtBlurY.texture;

    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    if (this.clear) renderer.clear();
    this._fsQuad.render(renderer);
  }

  setSize(w, h) {
    this._rtBright.setSize(w, h);
    this._rtBlurX.setSize(w, h);
    this._rtBlurY.setSize(w, h);
    this.blurMaterial.uniforms.resolution.value.set(w, h);
  }

  dispose() {
    this._rtBright.dispose();
    this._rtBlurX.dispose();
    this._rtBlurY.dispose();
    this.thresholdMaterial.dispose();
    this.blurMaterial.dispose();
    this.combineMaterial.dispose();
    this._fsQuad.dispose();
  }
}
