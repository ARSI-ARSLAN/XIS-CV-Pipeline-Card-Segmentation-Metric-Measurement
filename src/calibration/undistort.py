"""
Undistortion utilities built on the intrinsics saved by calibrate_camera.py.
This is the mandatory step between "raw capture" and "labelling / training /
measurement" -- see docs/measurement_methodology.md for why skipping it
biases every downstream millimetre measurement.
"""
import cv2
import numpy as np
import os
import glob
import argparse


def load_calibration(path):
    d = np.load(path)
    return d["camera_matrix"], d["dist_coeffs"], tuple(int(x) for x in d["image_size"])


def undistort_image(img, K, dist, alpha=0):
    """Undistort a single BGR image (numpy array). Returns (undistorted_img, new_K)."""
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha)
    und = cv2.undistort(img, K, dist, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y + rh, x:x + rw]
        newK = newK.copy()
        newK[0, 2] -= x
        newK[1, 2] -= y
    return und, newK


def undistort_folder(input_dir, output_dir, calib_path):
    K, dist, size = load_calibration(calib_path)
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(input_dir, "*.png")) +
                     glob.glob(os.path.join(input_dir, "*.jpg")))
    for fname in images:
        img = cv2.imread(fname)
        und, _ = undistort_image(img, K, dist)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(fname)), und)
    print(f"Undistorted {len(images)} images -> {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/dataset/images")
    ap.add_argument("--output", default="data/dataset/images_undistorted")
    ap.add_argument("--calib", default="calib/camera_calibration.npz")
    args = ap.parse_args()
    undistort_folder(args.input, args.output, args.calib)
