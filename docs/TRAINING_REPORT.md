# Training Report — Mask R-CNN Card Segmentation

## Architecture

**Model:** Mask R-CNN with ResNet-50-FPN backbone
**Source:** `torchvision.models.detection.maskrcnn_resnet50_fpn` (COCO-pretrained weights)
**Fine-tuned heads:** box predictor and mask predictor replaced for 2 classes (background + card)

### Architecture Justification

The assignment explicitly excludes YOLO-family models and Roboflow-hosted models. Mask R-CNN was selected for the following reasons:

| Reason | Detail |
|---|---|
| Direct mask output | Produces per-instance binary masks, not just bounding boxes — required by Step 3 for contour-based mm measurement |
| RoIAlign | Sub-pixel-accurate mask boundary extraction, critical for millimetre-precision edge detection |
| Transfer learning | COCO-pretrained backbone works well on small single-class datasets (83 images) |
| Architectural lineage | Two-stage detector (RPN → RoIAlign → box/mask heads) — entirely distinct from YOLO's single-stage anchor regression |
| Maturity | Well-documented, reproducible, standard torchvision implementation with no third-party dependencies |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 4 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 5 × 10⁻⁴ |
| Base learning rate | 0.005 |
| LR schedule | StepLR — step size 10 epochs, γ = 0.1 |
| Score threshold (inference) | 0.7 |
| Train / Val / Test images | 58 / 17 / 8 |
| Training hardware | GPU (Google Colab T4) |
| Input | Undistorted images (`dataset/images_undistorted/`) |

**Augmentation:** Standard torchvision pipeline (random horizontal flip). No additional augmentations applied — the dataset size and task simplicity (single rigid object, controlled backgrounds) did not require heavy augmentation.

Training script: `models/train_maskrcnn.py`
Colab notebook: `inference/colab_training_script.py`

---

## Training Loss Curves

Loss curve plot: `models/loss_curve.png`

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.5582 | 0.3515 |
| 5 | 0.2025 | 0.1657 |
| 10 | 0.1065 | 0.1221 |
| 15 | 0.0935 | 0.1123 |
| 20 | 0.0897 | 0.1105 |
| 25 | 0.0877 | 0.1131 |
| 30 | 0.0890 | 0.1112 |

The model converges rapidly in the first 10 epochs. From epoch 10 onward both losses plateau, indicating the model has learned the task. No divergence between train and val loss — no significant overfitting for this dataset size.

Full per-epoch values: `models/history.json`

---

## Evaluation Metrics

Evaluated on the held-out **test split (8 images)** using pycocotools COCO evaluation and custom mask IoU metrics.

### COCO mAP (Bounding Box)

| Metric | Value |
|---|---|
| mAP@0.5 | **1.000** |
| mAP@0.5:0.95 | **0.871** |

### COCO mAP (Segmentation Mask)

| Metric | Value |
|---|---|
| mAP@0.5 | **1.000** |
| mAP@0.5:0.95 | **0.918** |

### Custom Mask Metrics @ IoU threshold 0.5

| Metric | Value |
|---|---|
| Mean IoU | **0.936** |
| Precision | **1.000** |
| Recall | **1.000** |
| F1 Score | **1.000** |
| True Positives | 7 |
| False Positives | 0 |
| False Negatives | 0 |

Raw metrics file: `models/metrics.json`
Test predictions (COCO RLE format): `models/test_predictions.json`

### Interpretation

- Perfect precision and recall on the test set — the model detects the card in every image with no false positives.
- Mean IoU of 0.936 at threshold 0.5 indicates tight mask boundaries, well-suited to the millimetre-precision measurement task.
- mAP@0.5:0.95 of 0.918 (segm) confirms the model maintains quality at stricter IoU thresholds.
- The high metrics are consistent with the task characteristics: single class, rigid planar object, controlled capture conditions.

---

## Qualitative Results

Inference script produces annotated output images with mask overlay, contour, confidence score, and measured dimensions:

```bash
python -m src.inference.infer \
  --image path/to/raw.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --out inference/demo_output.jpg
```

Example annotated result: `measurement/result.jpg`

---

## Known Limitations

- Trained on a single-instance, single-class dataset — performance on scenes with multiple cards or heavy occlusion is not validated.
- Background diversity in training data is moderate (desk/table surfaces); highly cluttered backgrounds may reduce confidence scores.
- Model was fine-tuned on a dataset of 83 images — a larger and more diverse dataset would improve generalisation.
- Extreme perspective distortion (card nearly edge-on) is not represented in the training data.
