// Author: Riccardo Petrini
import {
  HalfFloatType,
  NearestFilter,
  NoBlending,
  ShaderMaterial,
  UniformsUtils,
  WebGLRenderTarget,
  Color
} from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const TintedAfterimageShader = {
  uniforms: {
    tNew: { value: null },
    tOld: { value: null },
    damp: { value: 0.9 },
    tint: { value: new Color(0.4, 0.8, 1.0) }
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tNew;
    uniform sampler2D tOld;
    uniform float damp;
    uniform vec3 tint;
    varying vec2 vUv;
    void main() {
      vec4 current = texture2D(tNew, vUv);
      vec4 trail = texture2D(tOld, vUv) * damp;
      vec3 color = mix(current.rgb, trail.rgb, damp);
      color = mix(color, color * tint, 0.25);
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

export class AfterimagePassTinted extends Pass {
  constructor(damp = 0.9, tint = new Color(0.4, 0.8, 1.0)) {
    super();
    this.uniforms = UniformsUtils.clone(TintedAfterimageShader.uniforms);
    this.uniforms.damp.value = damp;
    this.uniforms.tint.value.copy(tint);

    this.compMaterial = new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: TintedAfterimageShader.vertexShader,
      fragmentShader: TintedAfterimageShader.fragmentShader
    });

    this.copyMaterial = new ShaderMaterial({
      uniforms: { tDiffuse: { value: null } },
      vertexShader: /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: /* glsl */`uniform sampler2D tDiffuse; varying vec2 vUv; void main(){ gl_FragColor = texture2D(tDiffuse, vUv); }`,
      blending: NoBlending,
      depthTest: false,
      depthWrite: false
    });

    this._textureComp = new WebGLRenderTarget(window.innerWidth, window.innerHeight, {
      magFilter: NearestFilter,
      type: HalfFloatType
    });
    this._textureOld = this._textureComp.clone();

    this._compQuad = new FullScreenQuad(this.compMaterial);
    this._copyQuad = new FullScreenQuad(this.copyMaterial);
  }

  get damp() { return this.uniforms.damp.value; }
  set damp(v) { this.uniforms.damp.value = v; }

  set tint(c) { this.uniforms.tint.value.copy(c); }

  render(renderer, writeBuffer, readBuffer) {
    this.uniforms.tNew.value = readBuffer.texture;
    this.uniforms.tOld.value = this._textureOld.texture;

    renderer.setRenderTarget(this._textureComp);
    this._compQuad.render(renderer);

    this._copyQuad.material.uniforms.tDiffuse.value = this._textureComp.texture;

    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
      this._copyQuad.render(renderer);
    } else {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
      this._copyQuad.render(renderer);
    }

    const tmp = this._textureOld;
    this._textureOld = this._textureComp;
    this._textureComp = tmp;
  }

  setSize(w, h) {
    this._textureComp.setSize(w, h);
    this._textureOld.setSize(w, h);
  }

  dispose() {
    this._textureComp.dispose();
    this._textureOld.dispose();
    this.compMaterial.dispose();
    this.copyMaterial.dispose();
    this._compQuad.dispose();
    this._copyQuad.dispose();
  }
}
