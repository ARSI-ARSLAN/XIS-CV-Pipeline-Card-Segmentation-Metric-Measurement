"""
Evaluates a trained Mask R-CNN checkpoint on the held-out test split.
Reports mAP@0.5 and mAP@0.5:0.95 (via pycocotools, box + segm) plus
precision/recall/F1/mean-IoU at an IoU threshold of 0.5 (mask-based, single class).

Usage:
    python -m src.train.evaluate --model models/maskrcnn_last.pth
"""
import os
import json
import argparse
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from src.train.dataset import CocoSegDataset, collate_fn
from src.train.train_maskrcnn import build_model


def run_inference_to_coco(model, loader, device, score_thresh=0.5):
    results = []
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            outputs = model(imgs)
            for target, out in zip(targets, outputs):
                img_id = int(target["image_id"].item())
                for box, score, label, mask in zip(out["boxes"], out["scores"],
                                                     out["labels"], out["masks"]):
                    if score < score_thresh:
                        continue
                    m = (mask[0] > 0.5).cpu().numpy().astype(np.uint8)
                    rle = maskUtils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    x1, y1, x2, y2 = box.cpu().numpy()
                    results.append({
                        "image_id": img_id, "category_id": int(label.item()),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score.item()), "segmentation": rle,
                    })
    return results


def evaluate(data_dir, splits_dir, model_path, out_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = os.path.join(data_dir, "images_undistorted")
    test_ds = CocoSegDataset(images_dir, os.path.join(splits_dir, "test.json"))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False,
                                               collate_fn=collate_fn)

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    dt = run_inference_to_coco(model, test_loader, device)
    dt_path = os.path.join(out_dir, "test_predictions.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(dt_path, "w") as f:
        json.dump(dt, f)

    coco_gt = COCO(os.path.join(splits_dir, "test.json"))
    metrics = {}
    if len(dt) > 0:
        coco_dt = coco_gt.loadRes(dt_path)
        for iou_type in ["bbox", "segm"]:
            e = COCOeval(coco_gt, coco_dt, iou_type)
            e.evaluate()
            e.accumulate()
            e.summarize()
            metrics[iou_type] = {"mAP@0.5:0.95": float(e.stats[0]), "mAP@0.5": float(e.stats[1])}
    else:
        metrics["bbox"] = metrics["segm"] = {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}

    # Custom precision/recall/F1/mean-IoU at IoU 0.5 (single class, mask-based)
    tp = fp = fn = 0
    ious = []
    gt_by_img, dt_by_img = {}, {}
    for ann in coco_gt.dataset["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    for d in dt:
        dt_by_img.setdefault(d["image_id"], []).append(d)

    for img_id in coco_gt.getImgIds():
        gts = gt_by_img.get(img_id, [])
        preds = sorted(dt_by_img.get(img_id, []), key=lambda x: -x["score"])
        matched = set()
        for p in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                g_rle = coco_gt.annToRLE(g)
                iou = maskUtils.iou([p["segmentation"]], [g_rle], [0])[0][0]
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5:
                tp += 1
                matched.add(best_j)
                ious.append(best_iou)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    metrics["custom_iou0.5"] = {
        "precision": precision, "recall": recall, "f1": f1,
        "mean_IoU": mean_iou, "tp": tp, "fp": fp, "fn": fn,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/dataset")
    ap.add_argument("--splits_dir", default="data/dataset/splits")
    ap.add_argument("--model", default="models/maskrcnn_last.pth")
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()
    evaluate(args.data_dir, args.splits_dir, args.model, args.out_dir)
