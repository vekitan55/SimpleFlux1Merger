# Simplified Flux.1 Merge for ComfyUI

This extension provides expert-level control over Flux.1 model merging via ComfyUI.

## Included Node

### ExpertFlux1Merge

- Per-layer weight control (double_blocks, single_blocks, and top-level layers)
- Merge modes: standard, add_difference, train_difference
- Efficient memory use with bfloat16
- Suitable for high-precision model tuning

## Usage

1. Place this folder in your `ComfyUI/custom_nodes/` directory.
2. Launch ComfyUI and add the node from the Flux.1 category.
3. Connect three compatible models (A, B, C).
4. Adjust sliders and choose merge mode.

## License

MIT License
