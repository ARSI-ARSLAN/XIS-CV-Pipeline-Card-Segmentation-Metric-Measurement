"""
Generates synthetic checkerboard images with a known, hidden "true" lens
distortion baked in via cv2.projectPoints. This exists ONLY so the
calibration pipeline can be developed and unit-tested without a physical
camera. Replace this step with 20+ real photos of a printed checkerboard
taken from varied angles/distances with YOUR camera before your final
submission -- calibrate_camera.py does not care where the images came from.
"""
import cv2
import numpy as np
import os
import argparse


def make_checkerboard_images(n_images=25, out_dir="data/calibration_images",
                              squares=(10, 7), square_size=25.0,
                              image_size=(1280, 960), seed=42):
    os.makedirs(out_dir, exist_ok=True)

    # "True" intrinsics/distortion used only to synthesize images.
    # calibrate_camera.py must recover values close to these from the images alone.
    fx = fy = 900.0
    cx, cy = image_size[0] / 2, image_size[1] / 2
    K_true = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_true = np.array([-0.28, 0.12, 0.001, -0.0008, -0.02], dtype=np.float64)

    sq_w, sq_h = squares
    xs = np.arange(sq_w + 1) * square_size
    ys = np.arange(sq_h + 1) * square_size
    xs = xs - xs.mean()
    ys = ys - ys.mean()

    rng = np.random.default_rng(seed)
    made = 0
    attempts = 0
    while made < n_images and attempts < n_images * 4:
        attempts += 1
        rvec = rng.uniform(-0.5, 0.5, 3)
        rvec[2] = rng.uniform(-np.pi, np.pi)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(350, 700)])

        img = np.full((image_size[1], image_size[0], 3), 255, np.uint8)
        ok = True
        quads = []
        for r in range(sq_h):
            for c in range(sq_w):
                color = (0, 0, 0) if (r + c) % 2 == 0 else (255, 255, 255)
                corners3d = np.array([
                    [xs[c],     ys[r],     0],
                    [xs[c + 1], ys[r],     0],
                    [xs[c + 1], ys[r + 1], 0],
                    [xs[c],     ys[r + 1], 0],
                ], dtype=np.float64)
                pts2d, _ = cv2.projectPoints(corners3d, rvec, tvec, K_true, dist_true)
                pts2d = pts2d.reshape(-1, 2)
                if not np.isfinite(pts2d).all():
                    ok = False
                    break
                quads.append((pts2d.astype(np.int32), color))
            if not ok:
                break
        if not ok:
            continue

        for pts, color in quads:
            cv2.fillConvexPoly(img, pts, color)

        fname = os.path.join(out_dir, f"calib_{made:03d}.png")
        cv2.imwrite(fname, img)
        made += 1

    print(f"Generated {made} synthetic checkerboard images -> {out_dir}")
    print("Hidden ground-truth intrinsics used to synthesize (for sanity-check only):")
    print("K_true =\n", K_true)
    print("dist_true =", dist_true)
    return K_true, dist_true


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="data/calibration_images")
    args = ap.parse_args()
    make_checkerboard_images(args.n, args.out)
