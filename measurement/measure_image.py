"""
Single-command end-to-end demo required by Step 3:
    raw image -> undistort -> segmentation mask -> width(mm), height(mm), confidence

Supports two reference modes:
  - ArUco marker detected in the scene (preferred)
  - Self-reference from known card dimensions (fallback when no marker present)

Usage:
    python -m src.measurement.measure_image --image raw.jpg \
        --calib calib/camera_calibration.npz \
        --model models/maskrcnn_last.pth \
        --out result.jpg
"""
import argparse
import json
import cv2

from src.calibration.undistort import load_calibration, undistort_image
from src.train.train_maskrcnn import build_model
from src.measurement.measure import (compute_pixels_per_mm_aruco,
                                      compute_pixels_per_mm_self_reference,
                                      mask_to_dimensions)
import torch
import torchvision.transforms.functional as F
import numpy as np


def run(image_path, calib_path, model_path, marker_mm=40.0,
        known_width_mm=85.60, out_path="result.jpg", score_thresh=0.7):
    K, dist, _ = load_calibration(calib_path)
    raw = cv2.imread(image_path)
    und, _ = undistort_image(raw, K, dist)  # mandatory - see docs/measurement_methodology.md

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = model([F.to_tensor(rgb).to(device)])[0]

    best_mask, best_score = None, -1.0
    for score, mask in zip(out["scores"], out["masks"]):
        if score >= score_thresh and float(score) > best_score:
            best_score = float(score)
            best_mask = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)

    if best_mask is None:
        print("No detection above threshold.")
        return None

    # Try ArUco first, fall back to self-reference
    ratio, marker_corners = compute_pixels_per_mm_aruco(und, marker_mm)
    ref_mode = "aruco"
    if ratio is None:
        ratio, _, _, _ = compute_pixels_per_mm_self_reference(
            best_mask, known_width_mm)
        marker_corners = None
        ref_mode = "self_reference"

    width_mm, height_mm, box = mask_to_dimensions(best_mask, ratio)

    vis = und.copy()
    overlay = vis.copy()
    overlay[best_mask > 0] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    cv2.polylines(vis, [box.astype(np.int32)], True, (0, 255, 0), 2)
    if marker_corners is not None:
        cv2.polylines(vis, [marker_corners.astype(np.int32)], True, (255, 0, 0), 2)
    label = f"W:{width_mm:.1f}mm H:{height_mm:.1f}mm conf:{best_score:.2f}"
    cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)

    result = {"width_mm": round(width_mm, 2), "height_mm": round(height_mm, 2),
              "confidence": round(best_score, 3), "pixels_per_mm": round(ratio, 4),
              "reference_mode": ref_mode}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default="calib/camera_calibration.npz")
    ap.add_argument("--model", default="models/maskrcnn_last.pth")
    ap.add_argument("--marker_mm", type=float, default=40.0)
    ap.add_argument("--known_width_mm", type=float, default=85.60,
                    help="Known long-side dimension in mm (used when ArUco not detected)")
    ap.add_argument("--out", default="result.jpg")
    ap.add_argument("--score_thresh", type=float, default=0.7)
    args = ap.parse_args()
    run(args.image, args.calib, args.model, args.marker_mm,
        args.known_width_mm, args.out, args.score_thresh)
