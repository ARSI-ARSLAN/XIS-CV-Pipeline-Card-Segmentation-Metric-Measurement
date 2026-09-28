"""
Generates a synthetic labelled dataset of the target object (an ID-1 card,
85.60mm x 53.98mm) PLUS a co-planar reference marker of known size
(default 40mm square), rendered through the same lens-distortion model used
in make_calibration_images.py.

This stands in for "70+ real photos labelled in CVAT/Roboflow" so the
training/measurement code can be built and tested end-to-end. To use your
own object:
  1. Capture 70+ real photos with your calibrated camera.
  2. Label them in CVAT or Roboflow (polygon/mask, single class), export as
     COCO JSON (images/ + annotations.json in this same layout).
  3. Point split_dataset.py / train_maskrcnn.py at your real folder instead.

Output layout (COCO instance-segmentation format):
    data/dataset/images/*.jpg
    data/dataset/annotations.json   (category "card" only -- the reference
                                      marker is NOT a training label, it's a
                                      measurement aid, see docs/measurement_methodology.md)
"""
import cv2
import numpy as np
import os
import json
import argparse

CARD_W_MM = 85.60
CARD_H_MM = 53.98
REF_MM = 40.0
REF_COLOR_BGR = (255, 0, 255)  # magenta - chosen to be easy to key out


def synth_dataset(n=90, out_dir="data/dataset", image_size=(1280, 960), seed=7):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    fx = fy = 900.0
    cx, cy = image_size[0] / 2, image_size[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([-0.28, 0.12, 0.001, -0.0008, -0.02])

    rng = np.random.default_rng(seed)
    images, annotations = [], []
    ann_id = 1
    img_id = 1

    half_w, half_h = CARD_W_MM / 2, CARD_H_MM / 2
    card3d = np.array([[-half_w, -half_h, 0], [half_w, -half_h, 0],
                        [half_w, half_h, 0], [-half_w, half_h, 0]])

    hw_r = REF_MM / 2
    gx = half_w + 30 + hw_r  # place reference marker 30mm to the right of the card, same plane
    ref3d = np.array([[gx - hw_r, -hw_r, 0], [gx + hw_r, -hw_r, 0],
                       [gx + hw_r, hw_r, 0], [gx - hw_r, hw_r, 0]])

    attempts = 0
    W, H = image_size
    while img_id <= n and attempts < n * 6:
        attempts += 1
        rvec = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35),
                          rng.uniform(-np.pi, np.pi)])
        tvec = np.array([rng.uniform(-140, 60), rng.uniform(-90, 90), rng.uniform(300, 900)])

        card_pts, _ = cv2.projectPoints(card3d, rvec, tvec, K, dist)
        ref_pts, _ = cv2.projectPoints(ref3d, rvec, tvec, K, dist)
        card_pts = card_pts.reshape(-1, 2)
        ref_pts = ref_pts.reshape(-1, 2)

        both = np.vstack([card_pts, ref_pts])
        if not np.isfinite(both).all():
            continue
        if both[:, 0].min() < 5 or both[:, 0].max() > W - 5 or \
           both[:, 1].min() < 5 or both[:, 1].max() > H - 5:
            continue  # keep both objects fully in frame

        # --- render background ---
        bg_color = rng.integers(60, 220, 3).tolist()
        img = np.full((H, W, 3), bg_color, np.uint8)
        noise = rng.normal(0, 8, (H, W, 3))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for _ in range(rng.integers(2, 6)):
            pt1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            pt2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            col = tuple(int(c) for c in rng.integers(0, 255, 3))
            if rng.random() < 0.5:
                cv2.rectangle(img, pt1, pt2, col, -1)
            else:
                cv2.circle(img, pt1, int(rng.integers(10, 80)), col, -1)

        # --- render reference marker (not a training label) ---
        cv2.fillConvexPoly(img, ref_pts.astype(np.int32), REF_COLOR_BGR)

        # --- render card (the training label) ---
        card_color = tuple(int(c) for c in rng.integers(150, 255, 3))
        poly = card_pts.astype(np.int32)
        cv2.fillConvexPoly(img, poly, card_color)
        cv2.polylines(img, [poly], True, (30, 30, 30), 2)
        darker = tuple(int(c * 0.85) for c in card_color)
        cv2.line(img, tuple(poly[0]), tuple(poly[2]), darker, 1)

        fname = f"card_{img_id:04d}.jpg"
        cv2.imwrite(os.path.join(img_dir, fname), img)
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})

        x, y, w, h = cv2.boundingRect(poly)
        annotations.append({
            "id": ann_id, "image_id": img_id, "category_id": 1,
            "segmentation": [poly.flatten().tolist()],
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(w * h), "iscrowd": 0,
        })
        ann_id += 1
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "card", "supercategory": "object"}],
    }
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(coco, f)

    print(f"Generated {len(images)} labelled images -> {img_dir}")
    print(f"COCO annotations -> {os.path.join(out_dir, 'annotations.json')}")
    print(f"Reference marker: {REF_MM}mm square, color BGR={REF_COLOR_BGR} "
          f"(placed co-planar with the card, used only for pixel-to-mm conversion)")
    return coco


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--out", default="data/dataset")
    args = ap.parse_args()
    synth_dataset(args.n, args.out)
