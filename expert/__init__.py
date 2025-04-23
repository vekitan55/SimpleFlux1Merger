import torch
from comfy.model_patcher import ModelPatcher
import uuid
import os

class ExpertFlux1Merge:
    CATEGORY = "Flux.1"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        input_dict = {
            "required": {
                "merge_mode": (["standard", "add_difference", "train_difference"], {"default": "standard"}),
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "model_c": ("MODEL",),
            }
        }

        # img_in〜txt_inを先頭に追加
        for name in ["img_in", "time_in", "guidance_in", "vector_in", "txt_in"]:
            input_dict["required"][name] = ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1})

        # 各レイヤーを個別に制御するスライダー（0〜1）
        for i in range(19):
            input_dict["required"][f"double_{i:02}"] = ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1})
        for i in range(38):
            input_dict["required"][f"single_{i:02}"] = ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1})

        # final_layer のみ追加
        input_dict["required"]["final_layer"] = ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.1})

        return input_dict

    def run(self, merge_mode, model_a, model_b, model_c, **kwargs):
        double_weights = {f"double_blocks.{i}": kwargs[f"double_{i:02}"] for i in range(19)}
        single_weights = {
            f"single_blocks.{i}": kwargs[f"single_{i:02}"] for i in range(38)
        }

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

            weight = 1.0
            for k, v in double_weights.items():
                if k in key:
                    weight = v
                    break
            for k, v in single_weights.items():
                if k in key:
                    weight = v
                    break
            for name in ["img_in", "time_in", "guidance_in", "vector_in", "txt_in", "final_layer"]:
                if name in key:
                    weight = kwargs[name]
                    break

            if isinstance(a_tensor, torch.Tensor) and isinstance(b_tensor, torch.Tensor) and a_tensor.shape == b_tensor.shape:
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
    "ExpertFlux1Merge": ExpertFlux1Merge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpertFlux1Merge": "Expert Flux.1 Merge"
}

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
