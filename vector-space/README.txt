Author: Riccardo Petrini

Contents (each folder is a self-contained variant)
- hover/: 10x10 grid, hover to inspect, click/drag to nudge values, double-click to reset cells. Shows first 16 values on the side.
- orbit/: Radial bar wheel; drag inward/outward on a wedge to set its value, hover highlights the wedge, canvas-based.
- slider/: Ten banks of faders (10 sliders each). Drag to set, double-click to reset, shift-click for fine nudges, includes random/zero/spread presets.
- heatmap/: Canvas heatmap that follows the cursor; adjust radius, freeze the field, reset or re-center. Values spike under the cursor with smooth falloff.

Notes
- Each variant keeps the vector in memory and renders the first 16 components for quick inspection.
- Values are clamped to [-1, 1] and stored in a shared array per page; copy the array logic into your own pipeline to feed a GAN or API.
- To explore internal convolutional layers when tied to a backend, reuse the hook approach from GANs/inference_cached.py and map these UI vectors into your generator input.
