import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F


class CocoSegDataset(torch.utils.data.Dataset):
    """Minimal COCO-format instance segmentation dataset for torchvision's
    Mask R-CNN. Expects polygon segmentations (as produced by CVAT/Roboflow
    COCO export, or generate_synthetic_dataset.py)."""

    def __init__(self, images_dir, ann_json):
        with open(ann_json) as f:
            coco = json.load(f)
        self.images_dir = images_dir
        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = list(self.images.keys())
        self.anns_by_img = {}
        for a in coco["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        img = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        w, h = info["width"], info["height"]
        anns = self.anns_by_img.get(img_id, [])

        boxes, labels, masks = [], [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(a["category_id"])
            seg = a["segmentation"]
            if isinstance(seg, dict):
                from pycocotools import mask as maskUtils
                mask = maskUtils.decode(seg).astype(np.uint8)
            elif isinstance(seg, list):
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly_pts in seg:
                    poly = np.array(poly_pts, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly], 1)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
            masks.append(mask)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8) if masks else torch.zeros((0, h, w), dtype=torch.uint8)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes) else torch.zeros((0,))

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64),
        }
        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))
