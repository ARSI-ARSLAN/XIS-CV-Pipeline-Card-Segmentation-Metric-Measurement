# XIS CV Pipeline — Card Segmentation & Metric Measurement

End-to-end computer vision pipeline that **segments an ID-1 format card and measures its real-world width and height in millimetres** using a calibrated camera, Mask R-CNN instance segmentation, and a co-planar reference marker.

---

## What This System Does

1. **Camera calibration** — removes lens distortion from all images using a checkerboard-based intrinsic calibration (OpenCV)
2. **Dataset collection & labelling** — 83 real photographs of an ID-1 card, labelled in Roboflow as COCO instance segmentation masks
3. **Mask R-CNN training** — fine-tunes a ResNet-50-FPN Mask R-CNN (torchvision) on the undistorted, labelled dataset
4. **Pixel-to-mm measurement** — uses a co-planar 40 mm ArUco marker to derive a local pixels-per-mm ratio, applies it to the segmentation mask contour, and outputs width/height in mm
5. **Accuracy validation** — compares system output against ISO ground-truth dimensions over 90 samples, reporting MAE and MPE

---

## Key Results

| Stage | Result |
|---|---|
| Calibration reprojection error | 1.02 px (30 images, 9×7 checkerboard) |
| Segmentation mAP@0.5 (mask) | **1.000** |
| Segmentation mAP@0.5:0.95 (mask) | **0.918** |
| Segmentation mean IoU | **0.936** |
| Measurement MAE — width | 5.29 mm / 2.89 mm (excl. outliers) |
| Measurement MAE — height | 3.32 mm / 1.95 mm (excl. outliers) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run inference on a new image (model already trained)
python -m src.inference.infer \
  --image path/to/photo.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --out output.jpg

# 3. Get width & height in mm
python -m src.measurement.measure_image \
  --image path/to/photo.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --out measurement/result.jpg
```

Full installation and run instructions: [`docs/SETUP.md`](docs/SETUP.md)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: raw image                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — Undistortion                                          │
│  cv2.undistort() using K, dist from camera_calibration.npz     │
└────────────────────────────┬────────────────────────────────────┘
                             │
               ┌─────────────┴──────────────┐
               ▼                            ▼
┌──────────────────────────┐  ┌─────────────────────────────────┐
│  STEP 2a — Reference     │  │  STEP 2b — Segmentation          │
│  ArUco marker detection  │  │  Mask R-CNN (ResNet-50-FPN)      │
│  → pixels_per_mm ratio   │  │  → binary instance mask          │
└─────────────┬────────────┘  └────────────────┬────────────────┘
              │                                │
              └─────────────┬──────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — Measurement                                           │
│  cv2.minAreaRect() on mask contour                              │
│  width_mm  = max(w_px, h_px) / pixels_per_mm                   │
│  height_mm = min(w_px, h_px) / pixels_per_mm                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: annotated image + {width_mm, height_mm, confidence}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
calibration/
├── camera_calibration.npz      Saved intrinsics (K, dist, reprojection error)
├── calibrate_camera.py         Calibration script
├── undistort.py                Undistortion utility
└── IMG_20260927_*.jpg          30 checkerboard calibration images

dataset/
├── images/                     83 raw object photos
├── images_undistorted/         Undistorted versions (used for training)
├── splits/                     train / val / test COCO JSON splits
├── annotations.json            Full COCO instance segmentation annotations
├── aruco_marker_id0.png        Printable 40 mm reference marker
└── split_dataset.py            Dataset splitting script

models/
├── maskrcnn_last.pth           Trained Mask R-CNN weights
├── metrics.json                Evaluation metrics (mAP, IoU, P/R/F1)
├── history.json                Per-epoch train/val loss
├── loss_curve.png              Training loss plot
├── train_maskrcnn.py           Training script
├── evaluate.py                 Evaluation script
└── dataset.py                  PyTorch dataset class

inference/
├── infer.py                    Inference script (undistort → segment → annotate)
└── colab_training_script.py    Google Colab GPU training notebook

measurement/
├── measure.py                  Core pixel-to-mm conversion module
├── measure_image.py            End-to-end single-image measurement script
├── validate_accuracy.py        Accuracy validation against ground truth
├── accuracy_report.csv         Per-image measurement results (90 samples)
├── accuracy_report_summary.json MAE / MPE summary
└── result.jpg                  Example annotated measurement output

docs/
├── CALIBRATION_REPORT.md       Camera calibration method, parameters, error
├── DATASET_CARD.md             Dataset description, collection, splits
├── TRAINING_REPORT.md          Architecture, hyperparameters, metrics
├── MEASUREMENT_REPORT.md       Pixel-to-mm methodology, accuracy analysis
└── SETUP.md                    Installation and run instructions

src/                            Python package (importable modules)
configs/config.yaml             Central configuration file
requirements.txt                Python dependencies
README.md                       This file
```

---

## Object: ID-1 Card

The measurement target is a standard ISO/IEC 7810 ID-1 format card:

| Property | Value |
|---|---|
| Width | 85.60 mm |
| Height | 53.98 mm |
| Examples | Credit card, debit card, loyalty card, ID card |

**Why this object:** rigid, planar, zero cost, exact standardised ground-truth dimensions, sharp edges for clean segmentation and contour fitting.

---

## Model: Mask R-CNN (ResNet-50-FPN)

Architecture selected to satisfy the assignment requirement of a non-YOLO, non-Roboflow model. Key properties for this task:

- **Two-stage detector** (RPN + RoIAlign + mask head) — architecturally distinct from YOLO-family single-stage detectors
- **Per-instance binary masks** — required for contour-based millimetre measurement
- **RoIAlign** — sub-pixel-accurate mask boundaries for precise edge extraction
- **COCO-pretrained backbone** — effective transfer learning for a small 83-image dataset

---

## Documentation

| Document | Description |
|---|---|
| [`docs/CALIBRATION_REPORT.md`](docs/CALIBRATION_REPORT.md) | Calibration method, camera matrix, distortion coefficients, reprojection error |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Object choice, collection strategy, labelling, class distribution, splits |
| [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md) | Architecture justification, hyperparameters, loss curves, evaluation metrics |
| [`docs/MEASUREMENT_REPORT.md`](docs/MEASUREMENT_REPORT.md) | Pixel-to-mm derivation, why undistortion is mandatory, MAE/MPE accuracy analysis |
| [`docs/SETUP.md`](docs/SETUP.md) | Full installation, environment setup, step-by-step run instructions |
