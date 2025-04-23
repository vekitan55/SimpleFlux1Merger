# Simple Flux.1 Merger for ComfyUI

A custom ComfyUI node set for merging Flux.1-based models with intuitive control.  
This extension provides **both simplified group merging** and **expert per-layer control**, including support for advanced difference-based merge modes.

---

## 📦 Included Nodes

### 1. **SimplifiedFlux1Merge**
- Intuitive UI with 9 group-based sliders for merging double and single transformer blocks.
- Preserves control over top-level layers like `img_in`, `txt_in`, `final_layer`.
- Designed for quick and effective model blending.

### 2. **ExpertFlux1Merge**
- Fine-grained control for all 19 `double_blocks`, 38 `single_blocks`, and all top-level blocks (`img_in`, `txt_in`, etc.).
- Ideal for advanced users seeking precise model tuning.
- Includes interpolation-safe FP16/BF16 handling and memory optimization.

---

## 🧠 Supported Merge Modes

| Mode              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `standard`        | Linear interpolation between A and B (`A * w + B * (1-w)`)                  |
| `add_difference`  | Adds difference between B and A to A (`A + w * (B - A)`)                     |
| `train_difference`| Adds training diff from C to A (`A + w * (B - C)`), mimicking LoRA learning |

---

## 🛠 Installation

Clone or download this repository into your ComfyUI `custom_nodes/` directory:

```bash
git clone https://github.com/yourname/comfyui-simplified-flux1.git

---
## 🚀 Usage
Launch ComfyUI.

Add either:

Simplified Flux.1 Merge node

Expert Flux.1 Merge node

Connect 3 Flux.1-compatible models to model_a, model_b, and model_c.

Adjust sliders to define merge ratios.

Select merge mode and execute.

---
## 📌 Notes
Compatible with Flux.1 and Schnell variants (if matching architecture).

Requires sufficient GPU memory; optimized for BF16.

train_difference mode requires model C to match A/B in structure.

---
## 📜 License
MIT License

---
## 🙏 Credits
Based on Flux.1 architecture by Black Forest Labs.

Inspired by SuperMerger.

Developed for ComfyUI advanced model merging workflows.
