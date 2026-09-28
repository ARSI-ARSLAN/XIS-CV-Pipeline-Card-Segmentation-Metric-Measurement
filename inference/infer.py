"""
End-to-end inference: raw image -> undistort -> Mask R-CNN segmentation ->
annotated image with the detected mask overlaid.

Usage:
    python -m src.inference.infer --image path/to/raw.jpg \
        --calib calib/camera_calibration.npz \
        --model models/maskrcnn_last.pth \
        --out inference_output.jpg
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from src.calibration.undistort import load_calibration, undistort_image
from src.train.train_maskrcnn import build_model


def infer(image_path, calib_path, model_path, out_path="inference_output.jpg", score_thresh=0.7):
    K, dist, _ = load_calibration(calib_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    und, newK = undistort_image(img, K, dist)  # mandatory before any measurement/inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).to(device)
    with torch.no_grad():
        out = model([tensor])[0]

    vis = und.copy()
    best_mask, best_score = None, -1.0
    for score, mask in zip(out["scores"], out["masks"]):
        if score < score_thresh:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_mask = mask[0].cpu().numpy()

    if best_mask is not None:
        m = (best_mask > 0.5).astype(np.uint8)
        overlay = vis.copy()
        overlay[m > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        cv2.putText(vis, f"score={best_score:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(vis, "no detection above threshold", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Saved annotated inference result -> {out_path} (best_score={best_score:.3f})")
    return best_mask, best_score, newK, und


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default="calib/camera_calibration.npz")
    ap.add_argument("--model", default="models/maskrcnn_last.pth")
    ap.add_argument("--out", default="inference_output.jpg")
    ap.add_argument("--score_thresh", type=float, default=0.7)
    args = ap.parse_args()
    infer(args.image, args.calib, args.model, args.out, args.score_thresh)
