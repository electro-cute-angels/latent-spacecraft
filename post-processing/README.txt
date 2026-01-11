Author: Riccardo Petrini

Variants added (post-processing)
- afterImagePass/
  - AfterimagePassTinted.js: adds a tinted trail with damped accumulation.
  - AfterimagePassLumaClamp.js: clamps luma to keep trails bright but controlled.
- bloomPass/
  - BloomPassDual.js: dual-threshold bloom with separable blur and soft knee.
  - BloomPassAnamorphic.js: directional streak bloom with horizontal emphasis.
- filmPass/
  - FilmPassGrainRoll.js: grain + gentle frame roll + vignette, optional grayscale.
  - FilmPassChromatic.js: chromatic shift with grain and scanline modulation.
- scanPixelPass/
  - ScanPixelPassDiagonal.js: pixelated diagonal sweep with tail.
  - ScanPixelPassPulse.js: vertical band that pulses and reveals pixels.

Notes
- All passes use Three.js addons Pass/ShaderPass and keep uniforms exposed for runtime tweaks.
