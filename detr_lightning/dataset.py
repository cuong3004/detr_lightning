from torchvision.datasets.coco import CocoDetection
from pycocotools.coco import COCO
import torch 

def collate_fn(batch):
    imgs = []
    boxes = []
    categorys = []
    for img, ann in batch:
        # img, ann = batch
        if ann != []:
            for obj in ann:
                boxes.append(obj['bbox'])
                categorys.append(obj['category_id'])
        else:
            boxes.append([])
        
        imgs.append(img)
    
    imgs = torch.stack(imgs)


data_coco = CocoDetection('../../dataset_coco/train', "../../dataset_coco/train/_annotations.coco.json")


coco = COCO("../../dataset_coco/train/_annotations.coco.json")

a = coco.loadAnns(coco.getAnnIds(10))
# print(a)

coco.getAnnIds(id)

print(a)

print(data_coco[10])

# print(data_coco[1])

