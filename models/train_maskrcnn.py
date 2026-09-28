"""
Fine-tunes torchvision's Mask R-CNN (ResNet-50-FPN) on the labelled,
UNDISTORTED object dataset.

Architecture justification (assignment requires non-YOLO, non-Roboflow):
Mask R-CNN is a two-stage instance segmentation architecture (RPN + RoIAlign +
mask head) with mature, well-documented torchvision pretrained weights. It is
a good fit here because: (a) it directly outputs per-instance masks -- not
just boxes -- which the measurement stage needs; (b) RoIAlign gives accurate
mask boundaries, which matters for millimetre-precision edge measurement;
(c) transfer learning from COCO pretrained weights works well with a small
(90-image) single-class dataset like ours; (d) it is a distinct lineage from
YOLO-family detectors and is not a Roboflow model.

Usage:
    python -m src.train.train_maskrcnn --epochs 30 --batch_size 4 --lr 0.005
"""
import os
import json
import argparse
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.train.dataset import CocoSegDataset, collate_fn


def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def train(data_dir, splits_dir, out_dir="models", epochs=30, batch_size=4, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = os.path.join(data_dir, "images_undistorted")

    train_ds = CocoSegDataset(images_dir, os.path.join(splits_dir, "train.json"))
    val_ds = CocoSegDataset(images_dir, os.path.join(splits_dir, "val.json"))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate_fn, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, num_workers=2)

    model = build_model(num_classes=2).to(device)  # background + card
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    os.makedirs(out_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
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
                loss_dict = model(imgs, targets)  # train-mode loss computation, no grad step
                val_running += sum(loss_dict.values()).item()
        val_loss = val_running / max(1, len(val_loader))
        history["val_loss"].append(val_loss)
        lr_sched.step()
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(out_dir, "maskrcnn_last.pth"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Mask R-CNN training/validation loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    print("Training complete. Weights + logs saved to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/dataset")
    ap.add_argument("--splits_dir", default="data/dataset/splits")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()
    train(args.data_dir, args.splits_dir, args.out_dir, args.epochs, args.batch_size, args.lr)
