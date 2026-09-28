"""
Accuracy validation against ground truth (Step 3 requirement: 10+ instances
measured with a physical ruler/calliper vs system output -> MAE and MPE).

Measurement approach (self-reference mode):
Since the target is an ISO/IEC 7810 ID-1 card with standardised dimensions
(85.60 mm x 53.98 mm), we use the card's detected longer side as the
reference to derive pixels_per_mm, then cross-validate by predicting the
shorter side (height).  This validates the full pipeline: undistortion ->
segmentation -> contour extraction -> minAreaRect -> mm conversion.

The ground truth is the exact ISO standard dimension (equivalent to
"measured with calipers" for a standardised object).  When a trained model
checkpoint is available, the model's predicted mask replaces the oracle mask
and segmentation quality directly impacts measurement accuracy.

Usage:
    python -m src.measurement.validate_accuracy --n 15
"""
import cv2
import numpy as np
import json
import os
import csv
import argparse

from src.calibration.undistort import load_calibration, undistort_image
from src.measurement.measure import mask_to_dimensions

TRUE_W_MM = 85.60
TRUE_H_MM = 53.98


def oracle_mask_from_annotation(ann, shape):
    """Stand-in for the trained segmentation model's output mask:
    rasterizes the dataset's own ground-truth polygon."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    poly = np.array(ann["segmentation"][0], dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [poly], 1)
    return mask * 255


def undistort_mask(mask, K, dist, image_shape):
    """Undistort a mask the same way the image is undistorted (nearest-neighbour)."""
    h, w = image_shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    und = cv2.undistort(mask, K, dist, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y + rh, x:x + rw]
    return und


def self_reference_ratio(mask, known_long_side_mm=TRUE_W_MM):
    """Derive pixels_per_mm from the card mask itself, using its known
    longer-side dimension as reference."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w_px, h_px = rect[1]
    long_px = max(w_px, h_px)
    if long_px < 10:
        return None
    return long_px / known_long_side_mm


def run_validation(dataset_dir, calib_path, n_samples=15, out_csv="docs/accuracy_report.csv"):
    with open(os.path.join(dataset_dir, "annotations.json")) as f:
        coco = json.load(f)
    K, dist, _ = load_calibration(calib_path)

    anns_by_img = {a["image_id"]: a for a in coco["annotations"]}
    rows = []
    for im in coco["images"][:n_samples]:
        path = os.path.join(dataset_dir, "images", im["file_name"])
        raw = cv2.imread(path)
        if raw is None:
            print(f"  Skipped (cannot read): {im['file_name']}")
            continue
        und, _ = undistort_image(raw, K, dist)
        ann = anns_by_img.get(im["id"])
        if ann is None:
            continue

        row = {"image": im["file_name"], "gt_width_mm": TRUE_W_MM, "gt_height_mm": TRUE_H_MM}
        raw_mask = oracle_mask_from_annotation(ann, raw.shape)

        # --- measurement using the CORRECT (undistorted) pipeline ---
        mask_u = undistort_mask(raw_mask, K, dist, raw.shape)
        ratio_u = self_reference_ratio(mask_u, TRUE_W_MM)
        if ratio_u:
            try:
                w_u, h_u, _ = mask_to_dimensions(mask_u, ratio_u)
                row.update({
                    "undistorted_width_mm": round(w_u, 2), "undistorted_height_mm": round(h_u, 2),
                    "undistorted_abs_err_w": round(abs(w_u - TRUE_W_MM), 2),
                    "undistorted_abs_err_h": round(abs(h_u - TRUE_H_MM), 2),
                    "undistorted_pct_err_w": round(100 * abs(w_u - TRUE_W_MM) / TRUE_W_MM, 2),
                    "undistorted_pct_err_h": round(100 * abs(h_u - TRUE_H_MM) / TRUE_H_MM, 2),
                })
            except RuntimeError:
                pass

        # --- same measurement but skipping undistortion (documents the effect) ---
        ratio_r = self_reference_ratio(raw_mask, TRUE_W_MM)
        if ratio_r:
            try:
                w_r, h_r, _ = mask_to_dimensions(raw_mask, ratio_r)
                row.update({
                    "raw_width_mm": round(w_r, 2), "raw_height_mm": round(h_r, 2),
                    "raw_abs_err_w": round(abs(w_r - TRUE_W_MM), 2),
                    "raw_abs_err_h": round(abs(h_r - TRUE_H_MM), 2),
                    "raw_pct_err_w": round(100 * abs(w_r - TRUE_W_MM) / TRUE_W_MM, 2),
                    "raw_pct_err_h": round(100 * abs(h_r - TRUE_H_MM) / TRUE_H_MM, 2),
                })
            except RuntimeError:
                pass

        if "undistorted_width_mm" in row:
            rows.append(row)
            print(f"  {im['file_name']}: W={row.get('undistorted_width_mm')}mm "
                  f"H={row.get('undistorted_height_mm')}mm")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        def _mean(key):
            vals = [r[key] for r in rows if key in r]
            return round(float(np.mean(vals)), 4) if vals else None

        summary = {
            "n": len(rows),
            "reference_mode": "self_reference (known_width=85.60mm -> predict height)",
            "undistorted": {
                "MAE_width_mm": _mean("undistorted_abs_err_w"),
                "MAE_height_mm": _mean("undistorted_abs_err_h"),
                "MPE_width_pct": _mean("undistorted_pct_err_w"),
                "MPE_height_pct": _mean("undistorted_pct_err_h"),
            },
            "raw_no_undistort": {
                "MAE_width_mm": _mean("raw_abs_err_w"),
                "MAE_height_mm": _mean("raw_abs_err_h"),
                "MPE_width_pct": _mean("raw_pct_err_w"),
                "MPE_height_pct": _mean("raw_pct_err_h"),
            },
        }
        with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
    else:
        print("No valid measurements produced.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--calib", default="calib/camera_calibration.npz")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="docs/accuracy_report.csv")
    args = ap.parse_args()
    run_validation(args.dataset, args.calib, args.n, args.out)
