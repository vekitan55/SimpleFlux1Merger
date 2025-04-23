from .expert import NODE_CLASS_MAPPINGS as EXPERT
from .simplified import NODE_CLASS_MAPPINGS as SIMPLIFIED

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(EXPERT)
NODE_CLASS_MAPPINGS.update(SIMPLIFIED)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpertFlux1Merge": "Expert Flux.1 Merge",
    "SimplifiedFlux1Merge": "Simplified Flux.1 Merge"
}

import os
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
