# %%
from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrConfig 
import torch
from PIL import Image
import requests
import json
# %%
config = DetrConfig.from_json_file("detr-resnet-50/config.json")
# with open("detr-resnet-50/preprocessor_config.json", 'r') as f:
#     config_extract = json.load(f)
#     print(type(config_extract))
# %%

feature_extractor = DetrFeatureExtractor.from_json_file("detr-resnet-50/preprocessor_config.json")
model = DetrForObjectDetection(config)


# %%
# model = DetrForObjectDetection(config)

# %%
from transformers.models.detr.modeling_detr import DetrTimmConvEncoder
# model = DetrTimmConvEncoder(config)


backbone = DetrTimmConvEncoder(
            config.backbone, config.dilation, config.use_pretrained_backbone, config.num_channels
        )

# %%

out = backbone(torch.ones((1,3,800,800)), pixel_mask=torch.ones((1,800,800)))
# %%
print(out[3][0].shape)

# %%
