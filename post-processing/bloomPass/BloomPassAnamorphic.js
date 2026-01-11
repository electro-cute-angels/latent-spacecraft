// Author: Riccardo Petrini
// Anamorphic bloom: stronger horizontal blur for streaks
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
    threshold: { value: 0.9 }
  },
  vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float threshold;
    varying vec2 vUv;
    void main(){
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      float l = max(max(c.r,c.g),c.b);
      float mask = step(threshold, l);
      gl_FragColor = vec4(c * mask, 1.0);
    }
  `
};

const BlurShader = {
  uniforms: {
    tDiffuse: { value: null },
    direction: { value: new Vector2(1.0, 0.0) },
    resolution: { value: new Vector2(1024, 1024) },
    stretch: { value: 2.5 }
  },
  vertexShader: ThresholdShader.vertexShader,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2 direction;
    uniform vec2 resolution;
    uniform float stretch;
    varying vec2 vUv;
    void main(){
      vec2 texel = direction / resolution * vec2(stretch, 1.0);
      vec3 col = vec3(0.0);
      col += texture2D(tDiffuse, vUv - 3.0*texel).rgb * 0.08;
      col += texture2D(tDiffuse, vUv - 2.0*texel).rgb * 0.12;
      col += texture2D(tDiffuse, vUv - 1.0*texel).rgb * 0.20;
      col += texture2D(tDiffuse, vUv).rgb * 0.20;
      col += texture2D(tDiffuse, vUv + 1.0*texel).rgb * 0.20;
      col += texture2D(tDiffuse, vUv + 2.0*texel).rgb * 0.12;
      col += texture2D(tDiffuse, vUv + 3.0*texel).rgb * 0.08;
      gl_FragColor = vec4(col, 1.0);
    }
  `
};

const CombineShader = {
  uniforms: {
    tScene: { value: null },
    tBloom: { value: null },
    intensity: { value: 1.5 }
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

export class BloomPassAnamorphic extends Pass {
  constructor(threshold = 0.9, intensity = 1.5, stretch = 2.5) {
    super();
    this.thresholdMat = new ShaderMaterial({
      uniforms: UniformsUtils.clone(ThresholdShader.uniforms),
      vertexShader: ThresholdShader.vertexShader,
      fragmentShader: ThresholdShader.fragmentShader
    });
    this.thresholdMat.uniforms.threshold.value = threshold;

    this.blurMat = new ShaderMaterial({
      uniforms: UniformsUtils.clone(BlurShader.uniforms),
      vertexShader: BlurShader.vertexShader,
      fragmentShader: BlurShader.fragmentShader
    });
    this.blurMat.uniforms.stretch.value = stretch;

    this.combineMat = new ShaderMaterial({
      uniforms: UniformsUtils.clone(CombineShader.uniforms),
      vertexShader: CombineShader.vertexShader,
      fragmentShader: CombineShader.fragmentShader,
      blending: AdditiveBlending,
      transparent: true
    });
    this.combineMat.uniforms.intensity.value = intensity;

    this._rtThreshold = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
    this._rtBlurX = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
    this._rtBlurY = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
    this._fsQuad = new FullScreenQuad(null);
  }

  render(renderer, writeBuffer, readBuffer) {
    // threshold
    this._fsQuad.material = this.thresholdMat;
    this.thresholdMat.uniforms.tDiffuse.value = readBuffer.texture;
    renderer.setRenderTarget(this._rtThreshold);
    renderer.clear();
    this._fsQuad.render(renderer);

    // horizontal streak
    this._fsQuad.material = this.blurMat;
    this.blurMat.uniforms.tDiffuse.value = this._rtThreshold.texture;
    this.blurMat.uniforms.direction.value.set(1.0, 0.0);
    renderer.setRenderTarget(this._rtBlurX);
    renderer.clear();
    this._fsQuad.render(renderer);

    // vertical soften
    this.blurMat.uniforms.tDiffuse.value = this._rtBlurX.texture;
    this.blurMat.uniforms.direction.value.set(0.0, 1.0);
    renderer.setRenderTarget(this._rtBlurY);
    renderer.clear();
    this._fsQuad.render(renderer);

    // combine
    this._fsQuad.material = this.combineMat;
    this.combineMat.uniforms.tScene.value = readBuffer.texture;
    this.combineMat.uniforms.tBloom.value = this._rtBlurY.texture;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    if (this.clear) renderer.clear();
    this._fsQuad.render(renderer);
  }

  setSize(w, h) {
    this._rtThreshold.setSize(w, h);
    this._rtBlurX.setSize(w, h);
    this._rtBlurY.setSize(w, h);
    this.blurMat.uniforms.resolution.value.set(w, h);
  }

  dispose() {
    this._rtThreshold.dispose();
    this._rtBlurX.dispose();
    this._rtBlurY.dispose();
    this.thresholdMat.dispose();
    this.blurMat.dispose();
    this.combineMat.dispose();
    this._fsQuad.dispose();
  }
}
