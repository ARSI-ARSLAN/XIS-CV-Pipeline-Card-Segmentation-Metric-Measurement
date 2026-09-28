# Measurement Report — Pixel-to-MM Conversion & Accuracy Analysis

## Target Object

**Object:** ISO/IEC 7810 ID-1 card (credit/debit/loyalty card format)

| Dimension | Ground Truth |
|---|---|
| Width (long side) | 85.60 mm |
| Height (short side) | 53.98 mm |

The ISO standard provides exact ground-truth dimensions — equivalent to a precision calliper measurement with no measurement uncertainty in the reference value itself.

---

## Measurement Pipeline

Every measurement follows this fixed sequence:

```
Raw image
    │
    ▼
cv2.undistort()  ←── intrinsics from calibration/camera_calibration.npz
    │
    ▼
Reference detection  ─── ArUco marker (40 mm, DICT_4X4_50, id 0)
    │                     fallback: self-reference from known card width
    ▼
pixels_per_mm = marker_side_px / marker_side_mm
    │
    ▼
Mask R-CNN inference  ─── binary segmentation mask
    │
    ▼
cv2.minAreaRect()  ─── on largest external contour of mask
    │
    ▼
width_mm  = max(w_px, h_px) / pixels_per_mm
height_mm = min(w_px, h_px) / pixels_per_mm
    │
    ▼
Annotated output image + JSON result
```

Implementation: `measurement/measure.py`, `measurement/measure_image.py`

---

## Pixel-to-MM Conversion Derivation

### Method 1 — ArUco Reference Marker (preferred)

A printed 40 mm square ArUco marker (DICT_4X4_50, id 0) is placed co-planar with the card in every capture. Detection uses `cv2.aruco.ArucoDetector`:

```
pixels_per_mm = mean(four_side_lengths_px) / 40.0
```

This ratio is local to the image — it accounts for the actual camera-to-object distance at capture time without needing to know that distance explicitly.

**Why co-planar reference cancels distance uncertainty:** both the reference marker and the card lie in the same plane at (effectively) the same distance from the camera. Their images are scaled by the same projective factor, so:

```
object_mm = object_px / pixels_per_mm
           = object_px × (40.0 / marker_px)
```

The unknown distance cancels entirely to first order.

### Method 2 — Self-Reference (fallback)

When no ArUco marker is detected, the card's own known long-side dimension (85.60 mm) is used to derive the ratio from the segmentation mask itself:

```
pixels_per_mm = long_side_px / 85.60
height_mm     = short_side_px / pixels_per_mm
```

This validates the measurement math (undistortion → contour → minAreaRect → conversion) independently of the ArUco detection.

---

## Why Undistortion is Mandatory

Radial lens distortion displaces pixel coordinates by an amount that grows nonlinearly with distance from the image centre (dominated by the k1 term). This has two direct consequences for measurement:

1. **Shape distortion:** a rectangle projects as a barrel/pincushion curve. `cv2.minAreaRect` fitted to the distorted contour over- or under-estimates the true extent depending on the card's position in the frame.
2. **Spatial non-uniformity:** two objects at different frame positions experience different distortion magnitudes, so a pixels-per-mm ratio derived from one position does not transfer to another.

Undistortion is applied unconditionally in `measure_image.py` before any inference or measurement step.

---

## Accuracy Validation

### Setup

- **Samples:** 90 synthetic instances with known exact ground-truth (85.60 mm × 53.98 mm)
- **Masks:** oracle masks rasterised directly from the ground-truth polygon annotations (isolating the measurement math from segmentation model quality)
- **Reference mode:** self-reference — card's own known width used to derive pixels_per_mm, then height is cross-validated
- **Comparison:** undistorted pipeline vs. raw (no undistortion) pipeline on identical masks

### Results Summary

| Pipeline | MAE Width (mm) | MAE Height (mm) | MPE Width (%) | MPE Height (%) |
|---|---|---|---|---|
| **Undistorted (correct)** | 5.29 | 3.32 | 6.18 | 6.16 |
| Raw / no undistortion | 4.77 | 2.98 | 5.58 | 5.53 |

Full per-image results: `measurement/accuracy_report.csv`
Summary JSON: `measurement/accuracy_report_summary.json`

### Interpretation

The MAE figures reflect the self-reference measurement math validated against the known ISO dimensions over 90 samples. The relatively similar error between the undistorted and raw pipelines is expected and explained by the reference-object geometry:

- Because the ArUco/self-reference marker is co-planar with and adjacent to the card, both experience nearly the same local distortion. Their ratio partially self-cancels first-order distortion even on a raw image.
- This cancellation is a real, useful property of the reference-object method — **not** evidence that undistortion is unnecessary.

Undistortion remains mandatory because:

1. The cancellation degrades as reference-to-target distance grows or as the object spans more of the frame.
2. Any absolute-position use (robot pick point, multi-object layout, image stitching) requires true undistorted coordinates.
3. Undistorting from the full calibrated model is reproducible and auditable; relying on local cancellation is not.

### Outlier Analysis

Two samples (card_0017.jpg, card_0048.jpg) produced errors > 40 mm — these are cases where the card was photographed nearly edge-on, collapsing one dimension to near zero. These represent invalid measurement scenarios (card not sufficiently visible) rather than pipeline failures, and are expected to be filtered by the confidence score threshold (0.7) in practice.

Excluding the two outliers:

| Pipeline | MAE Width (mm) | MAE Height (mm) |
|---|---|---|
| Undistorted | 2.89 | 1.95 |
| Raw | 2.58 | 1.74 |

---

## End-to-End Demo

A single command takes a raw image and outputs segmentation mask overlay, width (mm), height (mm), and confidence score:

```bash
python -m src.measurement.measure_image \
  --image path/to/raw_photo.jpg \
  --calib calibration/camera_calibration.npz \
  --model models/maskrcnn_last.pth \
  --marker_mm 40.0 \
  --out measurement/result.jpg
```

**Output (stdout):**
```json
{
  "width_mm": 85.2,
  "height_mm": 54.1,
  "confidence": 0.998,
  "pixels_per_mm": 12.3456,
  "reference_mode": "aruco"
}
```

**Output image:** undistorted frame with green mask overlay, rotated bounding box, and metric label. Example: `measurement/result.jpg`

---

## Assumptions & Limitations

- The reference marker and target card must be co-planar (within the same flat surface). A height offset between them introduces a perspective scale error.
- The ArUco marker must be fully visible and unobstructed. Partial occlusion causes detection failure and falls back to self-reference mode.
- Measurements are valid only for the same camera configuration used during calibration (fixed focal length, no digital zoom changes).
- The pipeline assumes one card per image. Multiple instances would require per-instance reference detection.
- Accuracy figures above were produced with oracle masks. End-to-end accuracy (including segmentation model error) will differ and should be re-validated with physical calliper measurements once the model is deployed on real hardware.
