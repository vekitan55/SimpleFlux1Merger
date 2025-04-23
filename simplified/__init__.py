import torch
from comfy.model_patcher import ModelPatcher
import uuid
import os

class SimplifiedFlux1Merge:
    CATEGORY = "Flux.1"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interpolate_groups": ("BOOLEAN", {"default": False}),
                "merge_mode": (["standard", "add_difference", "train_difference"], {"default": "standard"}),
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "model_c": ("MODEL",),

                "group_00_01": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_02_03": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_04_05": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_06_07": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_08_09": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_10_11": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_12_13": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_14_15": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
                "group_16_18": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
            }
        }

    def run(self, merge_mode, model_a, model_b, model_c, interpolate_groups,
            group_00_01, group_02_03, group_04_05, group_06_07,
            group_08_09, group_10_11, group_12_13, group_14_15, group_16_18):

        weights_raw = {
            "group_00_01": group_00_01,
            "group_02_03": group_02_03,
            "group_04_05": group_04_05,
            "group_06_07": group_06_07,
            "group_08_09": group_08_09,
            "group_10_11": group_10_11,
            "group_12_13": group_12_13,
            "group_14_15": group_14_15,
            "group_16_18": group_16_18,
        }

        group_ranges = [
            range(0, 2), range(2, 4), range(4, 6), range(6, 8), range(8, 10),
            range(10, 12), range(12, 14), range(14, 16), range(16, 19)
        ]

        weights = {}
        keys = list(weights_raw.keys())
        if interpolate_groups:
            for idx, group in enumerate(group_ranges):
                w_start = weights_raw[keys[idx]]
                if idx + 1 < len(keys):
                    w_end = weights_raw[keys[idx + 1]]
                else:
                    w_end = w_start
                steps = len(group)
                for i, j in enumerate(group):
                    weights[j] = w_start + (w_end - w_start) * (i / (steps - 1)) if steps > 1 else w_start
        else:
            for idx, group in enumerate(group_ranges):
                for j in group:
                    weights[j] = weights_raw[keys[idx]]

        model_a_weights = dict(model_a.model.state_dict())
        model_b_weights = dict(model_b.model.state_dict())
        model_c_weights = dict(model_c.model.state_dict())


        merged_model = {"model": {}}

        for key in model_a_weights:
            if key == "model_sampling.sigmas":
                continue

            device = model_a.load_device
            a_tensor = model_a_weights[key].to(dtype=torch.bfloat16, device=device)
            b_tensor = model_b_weights.get(key, a_tensor).to(dtype=torch.bfloat16, device=device)
            c_tensor = model_c_weights.get(key, a_tensor).to(dtype=torch.bfloat16, device=device)

            if isinstance(a_tensor, torch.Tensor) and isinstance(b_tensor, torch.Tensor) and a_tensor.shape == b_tensor.shape:
                weight = 1.0
                for i, r in enumerate(group_ranges):
                    for j in r:
                        if f"double_blocks.{j}" in key or f"single_blocks.{j * 2}" in key or f"single_blocks.{j * 2 + 1}" in key:
                            weight = weights[j]
                            break

                if merge_mode == "add_difference":
                    diff = weight * (b_tensor - a_tensor)
                    merged_model["model"][key] = a_tensor + diff
                elif merge_mode == "train_difference":
                    diff = weight * (b_tensor - c_tensor)
                    merged_model["model"][key] = a_tensor + diff
                else:
                    merged_model["model"][key] = a_tensor * weight + b_tensor * (1 - weight)
            else:
                merged_model["model"][key] = a_tensor

            del a_tensor, b_tensor, c_tensor

        torch.cuda.empty_cache()

        del model_a_weights, model_b_weights, model_c_weights

        model_class = type(model_a.model)
        model_config = getattr(model_a.model, "model_config", None)
        new_model = model_class(model_config=model_config)
        new_model.load_state_dict(merged_model["model"], strict=False)

        patched = ModelPatcher(model=new_model, load_device=model_a.load_device, offload_device=model_a.offload_device)
        patched.model_hash = str(uuid.uuid4())

        return (patched,)

NODE_CLASS_MAPPINGS = {
    "SimplifiedFlux1Merge": SimplifiedFlux1Merge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimplifiedFlux1Merge": "Simplified Flux.1 Merge"
}

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
