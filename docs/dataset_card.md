# Dataset Card — ID-1 Card Segmentation Dataset

## Object Selection

**Object:** ISO/IEC 7810 ID-1 format card (standard credit/debit/loyalty card)

| Property | Value |
|---|---|
| Real-world width | 85.60 mm |
| Real-world height | 53.98 mm |
| Standard | ISO/IEC 7810 ID-1 |

**Justification for object choice:**
- Universally available — any expired card works, zero cost
- Perfectly rigid and planar — no depth ambiguity or deformation
- Internationally standardised dimensions — ground-truth size is exact, not measured
- Sharp rectangular edges make polygon labelling fast and unambiguous
- Known aspect ratio enables cross-validation of measurement accuracy without a separate reference

---

## Dataset Summary

| Property | Value |
|---|---|
| Object class | `card` (1 class) |
| Total images | 83 |
| Annotation format | COCO instance segmentation (polygon masks) |
| Annotation file | `dataset/annotations.json` |
| Image resolution | 4000 × 3000 px (captured), resized for training |
| Labelling tool | Roboflow (polygon / mask tool) |
| Export format | COCO JSON |

---

## Collection Strategy

- Images were captured with a single fixed camera (the same one used for calibration — no lens/zoom changes between sessions).
- The card was photographed across varied:
  - **Distances** — near, mid, far within the camera's depth of field
  - **Angles** — straight-on, mild tilts, rotations in-plane
  - **Backgrounds** — multiple surfaces (desk, floor, paper)
  - **Lighting** — indoor natural light, artificial light, mixed
- A printed 40 mm ArUco marker (`DICT_4X4_50`, id 0) was placed co-planar with the card in every shot — this serves as the pixel-to-mm reference at inference time and is **not** a training class.
- All raw images were undistorted using `calibration/undistort.py` before labelling. Labels were drawn on the undistorted images so annotation coordinates are in corrected pixel space.

---

## Labelling

| Property | Value |
|---|---|
| Tool | Roboflow |
| Annotation type | Polygon (instance segmentation mask) |
| Classes | 1 — `card` (category\_id = 1) |
| Instances per image | 1 |
| Labelling time | ~2 min/image |

Polygon vertices tightly follow the card boundary. The ArUco marker visible in each image was deliberately not labelled — it exists only as a measurement reference.

---

## Dataset Splits

Split ratio: **70 % train / 20 % val / 10 % test** (seeded shuffle, seed=0)

| Split | Images | Annotations |
|---|---|---|
| train | 58 | 58 |
| val | 17 | 17 |
| test | 8 | 8 |
| **Total** | **83** | **83** |

Split files: `dataset/splits/train.json`, `val.json`, `test.json`
Split summary: `dataset/splits/split_summary.json`

---

## Class Distribution

Single class (`card`, category_id = 1). One instance per image across all splits.

| Split | card instances |
|---|---|
| train | 58 |
| val | 17 |
| test | 8 |

---

## File Layout

```
dataset/
├── images/                  # Raw captured photos (83 images)
├── images_undistorted/      # Undistorted versions — used for training and measurement
├── splits/
│   ├── train.json           # COCO-format split
│   ├── val.json
│   ├── test.json
│   └── split_summary.json
├── annotations.json         # Full COCO instance segmentation annotations
├── aruco_marker_id0.png     # Printable 40 mm reference marker (DICT_4X4_50, id 0)
└── split_dataset.py         # Script to regenerate splits
```

---

## Assumptions & Limitations

- All images feature exactly one card instance — multi-card scenes are not represented.
- Background diversity is moderate; performance on highly cluttered or low-contrast backgrounds has not been validated.
- The dataset does not include partially occluded or severely perspective-distorted cards.
- Lighting variation is limited to indoor conditions; outdoor or harsh directional lighting was not captured.
