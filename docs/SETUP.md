# Setup & Installation Guide

## Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.9+ |
| GPU | Recommended for training (CUDA-capable). CPU works but training is slow. |
| RAM | 8 GB+ |
| Disk | ~3 GB (dataset + model weights) |

---

## 1. Clone & Environment

```bash
git clone <repository-url>
cd xis-cv-pipeline

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs: `opencv-contrib-python`, `torch`, `torchvision`, `pycocotools`, `numpy`, `matplotlib`, `Pillow`, `pandas`, `pyyaml`, `tqdm`.

> **Note:** For GPU training, install the CUDA-enabled PyTorch build matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## 2. Camera Calibration

### With your own camera

Print a 10×8 square checkerboard (9×7 inner corners, 25 mm square size). Capture 20+ photos from varied angles, distances, and orientations — keep the full board in frame and in sharp focus.

Place images in `calibration/` (alongside the existing `.jpg` files, or replace them), then run:

```bash
python -m src.calibration.calibrate_camera \
  --input calibration/ \
  --pattern_w 9 --pattern_h 7 \
  --square_size 25.0 \
  --output calibration/camera_calibration.npz \
  --report docs/CALIBRATION_REPORT.md
```

Check `docs/CALIBRATION_REPORT.md` — mean reprojection error should be < 0.5 px. If not, recapture with sharper images from more varied angles.

### Using the pre-computed calibration (default)

`calibration/camera_calibration.npz` is already present. Skip this step.

---

## 3. Collect & Undistort Object Dataset

Capture 70+ photos of the ID-1 card with the **same camera** used in step 2. Place a printed 40 mm ArUco marker (`DICT_4X4_50`, id 0 — printable file: `dataset/aruco_marker_id0.png`) co-planar with the card in every shot.

Place raw images in `dataset/images/`, then undistort them:

```bash
python -m src.calibration.undistort \
  --input dataset/images \
  --output dataset/images_undistorted \
  --calib calibration/camera_calibration.npz
```

### Using the pre-collected dataset (default)

`dataset/images/` and `dataset/images_undistorted/` are already populated. Skip collection and undistortion.

---

## 4. Label the Dataset

Label the **undistorted** images in [Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai):
- Single class: `card`
- Annotation type: polygon / instance segmentation mask
- Export format: COCO JSON

Save the exported file as `dataset/annotations.json`.

### Using the pre-labelled annotations (default)

`dataset/annotations.json` is already present. Skip this step.

---

## 5. Split the Dataset

```bash
python -m src.data.split_dataset \
  --ann dataset/annotations.json \
  --out dataset/splits
```

Outputs `train.json`, `val.json`, `test.json`, `split_summary.json` in `dataset/splits/`.

---

## 6. Train the Model

```bash
python -m src.train.train_maskrcnn \
  --data_dir dataset \
  --splits_dir dataset/splits \
  --out_dir models \
  --epochs 30 \
  --batch_size 4 \
  --lr 0.005
```

Outputs:
- `models/maskrcnn_last.pth` — saved weights
- `models/history.json` — per-epoch train/val loss
- `models/loss_curve.png` — loss curve plot

### Training on Google Colab (GPU)

Use `inference/colab_training_script.py` — copy each cell into a new Colab notebook and follow the in-file instructions. Download the trained model back to `models/maskrcnn_last.pth`.

### Using the pre-trained model (default)

`models/maskrcnn_last.pth` is already present. Skip training.

---

## 7. Evaluate the Model

```bash
python -m src.train.evaluate \
  --data_dir dataset \
  --splits_dir dataset/splits \
  --model models/maskrcnn_last.pth \
  --out_dir models
```

Outputs `models/metrics.json` and `models/test_predictions.json`.

---

## 8. Run Inference on a New Image

```bash
python -m src.inference.infer \
  --image path/to/raw_photo.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --out inference/demo_output.jpg
```

Produces an annotated image with the segmentation mask overlay and confidence score.

---

## 9. Measure Width & Height (End-to-End)

```bash
python -m src.measurement.measure_image \
  --image path/to/raw_photo.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --marker_mm 40.0 \
  --out measurement/result.jpg
```

Prints JSON to stdout:

```json
{
  "width_mm": 85.2,
  "height_mm": 54.1,
  "confidence": 0.998,
  "pixels_per_mm": 12.3456,
  "reference_mode": "aruco"
}
```

---

## 10. Accuracy Validation

```bash
python -m src.measurement.validate_accuracy \
  --dataset dataset \
  --calib calibration/camera_calibration.npz \
  --n 90 \
  --out measurement/accuracy_report.csv
```

Outputs `measurement/accuracy_report.csv` and `measurement/accuracy_report_summary.json` with MAE and MPE for both the undistorted and raw pipelines.

---

## Configuration

All key parameters live in `configs/config.yaml`:

```yaml
target_object:
  real_width_mm: 85.60
  real_height_mm: 53.98
  reference_marker_mm: 40.0

calibration:
  checkerboard_inner_corners: [9, 7]
  square_size_mm: 25.0

training:
  epochs: 30
  batch_size: 4
  lr: 0.005
```

---

## Repository Structure

```
calibration/          Camera calibration images, scripts, and saved .npz
dataset/              Raw images, undistorted images, annotations, splits
models/               Trained weights, metrics, loss curves, training scripts
inference/            Inference script and Colab training notebook
measurement/          Measurement scripts, accuracy report, demo result
docs/                 All documentation files
src/                  Python package (importable modules)
configs/config.yaml   Central configuration
requirements.txt      Python dependencies
README.md             Project overview and quick-start
```
