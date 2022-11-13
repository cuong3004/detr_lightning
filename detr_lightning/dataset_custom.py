# %%
from torchvision.datasets.coco import CocoDetection
from pycocotools.coco import COCO
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
import numpy as np
# %%

def transform_box(boxes, w, h):
    # print(boxes)
    # print(boxes.shape)
    boxes = boxes * np.array([1/w, 1/h, 1/w, 1/h])
    return boxes
    
def collate_fn(batch):
    imgs = []
    boxes = []
    categorys = []
    for img, ann in batch:
        
        w, h = img.size
        # img, ann = batch
        if ann != []:
            box_obj = []
            for obj in ann:
                box_obj.append(obj['bbox'])
                categorys.append(obj['category_id'])
            box_obj = np.asarray(box_obj)
            box_obj = transform_box(box_obj, w, h)
            boxes.append(box_obj)
        else:
            boxes.append([])
        
        img = transforms(img)
        
        imgs.append(img)
    
    imgs = torch.stack(imgs)
    
    return imgs, boxes, categorys
# %%
transforms = trans.Compose([
    trans.ToTensor(),
    trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
from IPython.display import display
from PIL import Image 
# img = Image.open("image.jpg")
# print()

# display(img)
# transforms(img)
# %%

# data = np.ones((2, 4))
# data * [2,3,4,5]
# %%
# transform_

# def transforms_map(images, targets):
    
    
#     # print(images)
    
#     # print(targets)
#     if len(targets) != 0:
#         print(targets[0])
#         targets[0] = targets[0] * [1/w, 1,h, 1/w, 1/h]

#     return , targets

data_coco = CocoDetection(
    '../../dataset_coco/train', 
    "../../dataset_coco/train/_annotations.coco.json",
    # transforms=transforms_map,    
)

# x, y = data_coco[2]
# display(x)
# print(y)
# %%

dataloader = DataLoader(data_coco, 
                        batch_size=5, 
                        shuffle=True, 
                        num_workers=2, 
                        pin_memory=True, 
                        drop_last=True, 
                        collate_fn=collate_fn)

# x, *y = next(iter(dataloader))
# print(x.shape)
# print(y[0])
# print(y[0][0][0])


# import matplotlib.pyplot as plt
# plt.imshow(x[0].permute(1,2,0).numpy())
# plt.scatter(y[0][0][0][0]*320, y[0][0][0][1]*320)
# plt.scatter(y[0][0][0][0]*320+y[0][0][0][2]*320, y[0][0][0][1]*320+y[0][0][0][3]*320)




# %%

# coco = COCO("../../dataset_coco/train/_annotations.coco.json")

# a = coco.loadAnns(coco.getAnnIds(10))
# # print(a)

# coco.getAnnIds(id)

# print(a)

# print(data_coco[10])


# print(x)
# print(y)
# print(data_coco[1])


# %%
