from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management
import numpy as np
import torch
from einops import rearrange

class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100})
        )

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Map"

    def execute(self, image, mask_bbox_padding=30, resolution=512, **kwargs):
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = MeshGraphormerDetector.from_pretrained("hr16/ControlNet-HandRefiner-pruned", cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding)
            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask[:, :, :1].astype(np.float32) / 255.0))
        return torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
    
NODE_CLASS_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": Mesh_Graphormer_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": "Mesh Graphormer - Hand Depth Map"
}