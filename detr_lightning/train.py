from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrConfig 
import torch
from PIL import Image
import requests
import json
from detr_lightning.dataset_custom import CocoDetection

config = DetrConfig.from_json_file("config.json")

feature_extractor = DetrFeatureExtractor.from_json_file("preprocessor_config.json")
# feature_extractor.size = 320

train_dataset = CocoDetection(img_folder='/content/dataset_caries/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder='/content/dataset_caries/valid', feature_extractor=feature_extractor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

from torch.utils.data import DataLoader

def collate_fn(batch):
    # print(batch[0][0].shape)
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)


import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
         # replace COCO classification head with custom head
        self.model = DetrForObjectDetection(config)

        #  self.model = DetrForObjectDetection.from_pretrained("hf-internal-testing/tiny-detr-mobilenetsv3", 
        #                                                      num_labels=len(id2label),
        #                                                      ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs
     
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
    

model = Detr(lr=1e-4, lr_backbone=1e-4, weight_decay=1e-4)

from pytorch_lightning import Trainer

trainer = Trainer(gpus=1)
trainer.fit(model)