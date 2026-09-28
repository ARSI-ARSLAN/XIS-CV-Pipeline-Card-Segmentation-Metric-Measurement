"""
Intrinsic camera calibration from checkerboard images (OpenCV).

Usage:
    python -m src.calibration.calibrate_camera \
        --input data/calibration_images \
        --pattern_w 9 --pattern_h 6 --square_size 25.0 \
        --output calib/camera_calibration.npz \
        --report docs/calibration_report.md
"""
import cv2
import numpy as np
import glob
import os
import argparse


def calibrate(input_dir, pattern_size=(9, 6), square_size=25.0,
              output_path="calib/camera_calibration.npz",
              report_path="docs/calibration_report.md"):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints, imgpoints, used_images = [], [], []
    images = sorted(glob.glob(os.path.join(input_dir, "*.png")) +
                     glob.glob(os.path.join(input_dir, "*.jpg")))
    if len(images) < 20:
        print(f"WARNING: only {len(images)} images found; 20+ recommended by the assignment spec.")

    img_shape = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            used_images.append(fname)
        else:
            rev_pattern = (pattern_size[1], pattern_size[0])
            found_rev, corners_rev = cv2.findChessboardCorners(
                gray, rev_pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found_rev:
                objp_rev = np.zeros((rev_pattern[0] * rev_pattern[1], 3), np.float32)
                objp_rev[:, :2] = np.mgrid[0:rev_pattern[0], 0:rev_pattern[1]].T.reshape(-1, 2) * square_size
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners_rev, (11, 11), (-1, -1), criteria)
                objpoints.append(objp_rev)
                imgpoints.append(corners2)
                used_images.append(fname)

    if len(objpoints) < 10:
        raise RuntimeError(
            f"Only {len(objpoints)} valid checkerboard detections out of {len(images)} images. "
            "Need more/better calibration images (varied angle, in focus, board fully visible).")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    per_image_errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = float(np.mean(np.linalg.norm(imgpoints[i].reshape(-1, 2) - projected.reshape(-1, 2), axis=1)))
        per_image_errors.append(error)
    mean_error = float(np.mean(per_image_errors))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, camera_matrix=K, dist_coeffs=dist,
             image_size=np.array(img_shape), reprojection_error=mean_error,
             n_images=len(objpoints))

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Camera Calibration Report\n\n")
        f.write(f"- Calibration images used: **{len(objpoints)} / {len(images)}** captured\n")
        f.write(f"- Checkerboard: {pattern_size[0]}x{pattern_size[1]} inner corners, "
                f"{square_size} mm squares\n")
        f.write(f"- Image resolution: {img_shape[0]} x {img_shape[1]}\n")
        f.write(f"- **Mean reprojection error: {mean_error:.4f} px**\n")
        verdict = "EXCELLENT (< 0.3 px)" if mean_error < 0.3 else (
            "ACCEPTABLE (< 0.5 px)" if mean_error < 0.5 else "ABOVE TARGET (>= 0.5 px, recapture recommended)")
        f.write(f"  - Verdict: {verdict}\n\n")
        f.write("## Camera matrix (K)\n```\n" + str(K) + "\n```\n\n")
        f.write("## Distortion coefficients (k1, k2, p1, p2, k3)\n```\n" +
                str(dist.ravel()) + "\n```\n\n")
        f.write("## Per-image reprojection error (px)\n\n")
        f.write("| Image | Error (px) |\n|---|---|\n")
        for fn, e in zip(used_images, per_image_errors):
            f.write(f"| {os.path.basename(fn)} | {e:.4f} |\n")

    print(f"Calibration complete. Mean reprojection error: {mean_error:.4f} px")
    print(f"Saved calibration -> {output_path}")
    print(f"Saved report      -> {report_path}")
    return K, dist, mean_error


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/calibration_images")
    ap.add_argument("--pattern_w", type=int, default=9, help="inner corners, width")
    ap.add_argument("--pattern_h", type=int, default=6, help="inner corners, height")
    ap.add_argument("--square_size", type=float, default=25.0, help="mm")
    ap.add_argument("--output", default="calib/camera_calibration.npz")
    ap.add_argument("--report", default="docs/calibration_report.md")
    args = ap.parse_args()
    calibrate(args.input, (args.pattern_w, args.pattern_h), args.square_size,
              args.output, args.report)
