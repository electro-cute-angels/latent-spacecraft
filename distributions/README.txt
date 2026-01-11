Author: Riccardo Petrini

Distributions (JS generators)
- uniform.js: sampleUniform(n, min, max)
- normal.js: sampleNormal(n, mean, std)
- lognormal.js: sampleLogNormal(n, mu, sigma)
- exponential.js: sampleExponential(n, lambda)
- beta.js: sampleBeta(n, alpha, beta)
- mixture.js: sampleMixture(n, weight, a, b) mixes two normals

Notes
- Beta uses a small gamma sampler (Marsaglia-Tsang) and will be slower than others.
- All return plain arrays; reshape or normalize downstream.
