// Author: Riccardo Petrini
import {
  HalfFloatType,
  NearestFilter,
  ShaderMaterial,
  UniformsUtils,
  WebGLRenderTarget
} from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const LumaAfterimageShader = {
  uniforms: {
    tNew: { value: null },
    tOld: { value: null },
    damp: { value: 0.93 },
    lumaFloor: { value: 0.08 },
    lumaCeil: { value: 1.2 }
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D tNew;
    uniform sampler2D tOld;
    uniform float damp;
    uniform float lumaFloor;
    uniform float lumaCeil;
    varying vec2 vUv;
    float luma(vec3 c){ return dot(c, vec3(0.299,0.587,0.114)); }
    void main(){
      vec3 fresh = texture2D(tNew, vUv).rgb;
      vec3 trail = texture2D(tOld, vUv).rgb * damp;
      vec3 mixc = mix(fresh, trail, damp);
      float l = clamp(luma(mixc), lumaFloor, lumaCeil);
      mixc *= smoothstep(lumaFloor, lumaCeil, l);
      gl_FragColor = vec4(mixc, 1.0);
    }
  `
};

export class AfterimagePassLumaClamp extends Pass {
  constructor(damp = 0.93, lumaFloor = 0.08, lumaCeil = 1.2) {
    super();
    this.uniforms = UniformsUtils.clone(LumaAfterimageShader.uniforms);
    this.uniforms.damp.value = damp;
    this.uniforms.lumaFloor.value = lumaFloor;
    this.uniforms.lumaCeil.value = lumaCeil;

    this.material = new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: LumaAfterimageShader.vertexShader,
      fragmentShader: LumaAfterimageShader.fragmentShader
    });

    this._textureComp = new WebGLRenderTarget(window.innerWidth, window.innerHeight, {
      magFilter: NearestFilter,
      type: HalfFloatType
    });
    this._textureOld = this._textureComp.clone();

    this._fsQuad = new FullScreenQuad(this.material);
  }

  render(renderer, writeBuffer, readBuffer) {
    this.uniforms.tNew.value = readBuffer.texture;
    this.uniforms.tOld.value = this._textureOld.texture;

    renderer.setRenderTarget(this._textureComp);
    this._fsQuad.render(renderer);

    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
      this._fsQuad.render(renderer);
    } else {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
      this._fsQuad.render(renderer);
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
    this.material.dispose();
    this._fsQuad.dispose();
  }
}
