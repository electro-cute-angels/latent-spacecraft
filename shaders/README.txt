Author: Riccardo Petrini

Noise shaders (Three.js TSL)
- noise01-value.js: Value noise, fast and blocky-friendly.
- noise02-perlin.js: Classic Perlin gradients.
- noise03-simplex.js: Lightweight simplex-style approximation.
- noise04-worley.js: Worley (cell) distance to nearest feature.
- noise05-fbm.js: Fractal Brownian motion stacking value noise octaves.
- noise06-ridged.js: Ridged multifractal using abs of fbm.
- noise07-cellular-blend.js: Worley/Perlin blend for organic cells.
- noise08-flow.js: Flow-advected fbm for streaking motion.

Usage
Each file exports make*Material(), returning a MeshBasicNodeMaterial with the noise colorNode wired. Import the helper you need and assign the material to a mesh to preview the pattern.
