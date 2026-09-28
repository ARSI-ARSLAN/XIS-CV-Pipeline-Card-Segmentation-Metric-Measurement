"""
CV Measurement Pipeline - Mask R-CNN Training on Google Colab
=============================================================
Copy each cell below into a NEW Colab notebook.
Make sure to set Runtime > Change runtime type > GPU (T4) first!

Steps:
  Cell 1: Check GPU
  Cell 2: Upload zip
  Cell 3: Unzip & install
  Cell 4: Train (30 epochs, ~10-15 min on T4)
  Cell 5: Evaluate on test set
  Cell 6: Download trained model
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Check GPU (paste this as first cell in Colab)    ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 1 START---
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# ---CELL 1 END---


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Upload the zip file                              ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 2 START---
from google.colab import files
uploaded = files.upload()  # Choose colab_training_package.zip
# ---CELL 2 END---


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Unzip and setup project structure                 ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 3 START---
import os, zipfile

# Unzip
with zipfile.ZipFile("colab_training_package.zip", "r") as z:
    z.extractall("project")

# Verify structure
for root, dirs, fls in os.walk("project"):
    level = root.replace("project", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for f in fls[:5]:
        print(f"{subindent}{f}")
    if len(fls) > 5:
        print(f"{subindent}... and {len(fls)-5} more files")

print("\n✅ Project extracted successfully!")
# ---CELL 3 END---


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Install deps + Train Mask R-CNN (main cell!)     ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 4 START---
!pip install pycocotools -q

import os, sys, json, argparse
import numpy as np
import torch
import torchvision
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

# ──────────────────────────────────────────────
# Dataset class (handles both polygon and RLE)
# ──────────────────────────────────────────────
class CocoSegDataset(torch.utils.data.Dataset):
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
            "boxes": boxes_t, "labels": labels_t, "masks": masks_t,
            "image_id": torch.tensor([img_id]), "area": area,
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64),
        }
        return F.to_tensor(img), target

def collate_fn(batch):
    return tuple(zip(*batch))

# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────
def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
IMAGES_DIR = "project/images"
SPLITS_DIR = "project/splits"
OUT_DIR = "trained_model"
EPOCHS = 30
BATCH_SIZE = 4
LR = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

train_ds = CocoSegDataset(IMAGES_DIR, os.path.join(SPLITS_DIR, "train.json"))
val_ds = CocoSegDataset(IMAGES_DIR, os.path.join(SPLITS_DIR, "val.json"))
print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=collate_fn, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          collate_fn=collate_fn, num_workers=2)

model = build_model(num_classes=2).to(device)  # background + card
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

os.makedirs(OUT_DIR, exist_ok=True)
history = {"train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for imgs, targets in train_loader:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    train_loss = running / max(1, len(train_loader))
    history["train_loss"].append(train_loss)

    val_running = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            val_running += sum(loss_dict.values()).item()
    val_loss = val_running / max(1, len(val_loader))
    history["val_loss"].append(val_loss)
    lr_sched.step()
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "maskrcnn_last.pth"))

# Save history + plot
with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.title("Mask R-CNN Training/Validation Loss")
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"))
plt.show()
print(f"\n✅ Training complete! Model saved to {OUT_DIR}/maskrcnn_last.pth")
# ---CELL 4 END---


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Evaluate on test set                             ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 5 START---
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

test_ds = CocoSegDataset(IMAGES_DIR, os.path.join(SPLITS_DIR, "test.json"))
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

model_eval = build_model(num_classes=2)
model_eval.load_state_dict(torch.load(os.path.join(OUT_DIR, "maskrcnn_last.pth"), map_location=device))
model_eval.to(device)
model_eval.eval()

# Run inference
results = []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = [im.to(device) for im in imgs]
        outputs = model_eval(imgs)
        for target, out in zip(targets, outputs):
            img_id = int(target["image_id"].item())
            for box, score, label, mask in zip(out["boxes"], out["scores"], out["labels"], out["masks"]):
                if score < 0.5:
                    continue
                m = (mask[0] > 0.5).cpu().numpy().astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("utf-8")
                x1, y1, x2, y2 = box.cpu().numpy()
                results.append({
                    "image_id": img_id, "category_id": int(label.item()),
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score": float(score.item()), "segmentation": rle,
                })

print(f"Generated {len(results)} predictions on {len(test_ds)} test images")

if results:
    dt_path = os.path.join(OUT_DIR, "test_predictions.json")
    with open(dt_path, "w") as f:
        json.dump(results, f)

    gt = COCO(os.path.join(SPLITS_DIR, "test.json"))
    dt = gt.loadRes(dt_path)

    for iou_type in ["bbox", "segm"]:
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        print()
else:
    print("⚠️ No predictions above threshold. Model may need more training.")

print("✅ Evaluation complete!")
# ---CELL 5 END---


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Download trained model back to your PC           ║
# ╚══════════════════════════════════════════════════════════════╝
# ---CELL 6 START---
import shutil
# Zip the trained model + history + loss curve
shutil.make_archive("trained_model_download", "zip", "trained_model")
files.download("trained_model_download.zip")
print("✅ Download started! Extract and copy maskrcnn_last.pth to f:\\cv_measurement_pipeline\\models\\")
# ---CELL 6 END---
