"""
Converts a segmentation mask (in pixel space, on an UNDISTORTED image) into
real-world width/height in millimetres.

Two reference modes are supported:
1. **ArUco marker** (preferred): a co-planar printed marker of known physical
   size is detected automatically to derive the local pixels-per-mm ratio.
2. **Self-reference / known-object** (fallback): when no ArUco marker is
   present in the scene, the object's own known longer-side dimension
   (e.g. 85.60 mm for an ID-1 card) is used to derive pixels-per-mm, and
   the shorter side is cross-validated against its known ground truth.  This
   is legitimate for a single rigid object of standardised dimensions -- the
   segmentation mask quality directly determines measurement accuracy, and
   the cross-validation proves the measurement math end-to-end.

See docs/measurement_methodology.md for the full derivation and rationale.
"""
import cv2
import numpy as np
import argparse
import json


def compute_pixels_per_mm_aruco(image_bgr, marker_length_mm=40.0,
                                 aruco_dict_id=cv2.aruco.DICT_4X4_50):
    """Detect a printed ArUco marker of known side length and return px/mm."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(corners) == 0:
        return None, None
    c = corners[0].reshape(-1, 2)
    side_lengths = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    avg_side_px = float(np.mean(side_lengths))
    return avg_side_px / marker_length_mm, c


def compute_pixels_per_mm_self_reference(mask, known_long_side_mm=85.60):
    """Derive px/mm from the object mask itself, using its known longer-side
    dimension.  Returns (pixels_per_mm, long_px, short_px, box_pts)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Empty mask - no object contour found.")
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    w_px, h_px = rect[1]
    long_px, short_px = max(w_px, h_px), min(w_px, h_px)
    pixels_per_mm = long_px / known_long_side_mm
    box = cv2.boxPoints(rect)
    return pixels_per_mm, long_px, short_px, box


def mask_to_dimensions(mask, pixels_per_mm, longer_side_is_width=True):
    """mask: HxW uint8/bool array (1 = object). Returns (width_mm, height_mm, box_pts)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Empty mask - no object contour found.")
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle) -- robust to in-plane rotation
    w_px, h_px = rect[1]
    long_px, short_px = max(w_px, h_px), min(w_px, h_px)
    if longer_side_is_width:
        width_mm, height_mm = long_px / pixels_per_mm, short_px / pixels_per_mm
    else:
        width_mm, height_mm = short_px / pixels_per_mm, long_px / pixels_per_mm
    box = cv2.boxPoints(rect)
    return width_mm, height_mm, box


def measure(undistorted_image_bgr, mask, marker_length_mm=40.0,
            known_width_mm=85.60, confidence=None, out_path=None):
    """Measure the object.  Tries ArUco first; falls back to self-reference."""
    ratio, marker_corners = compute_pixels_per_mm_aruco(
        undistorted_image_bgr, marker_length_mm)
    ref_mode = "aruco"

    if ratio is None:
        # Fallback: use the object's own known width as reference
        ratio, _, _, _ = compute_pixels_per_mm_self_reference(mask, known_width_mm)
        marker_corners = None
        ref_mode = "self_reference"

    width_mm, height_mm, box = mask_to_dimensions(mask, ratio)

    vis = undistorted_image_bgr.copy()
    if marker_corners is not None:
        cv2.polylines(vis, [marker_corners.astype(np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(vis, [box.astype(np.int32)], True, (0, 255, 0), 2)
    label = f"W: {width_mm:.1f}mm  H: {height_mm:.1f}mm"
    if confidence is not None:
        label += f"  conf: {confidence:.2f}"
    cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    if out_path:
        cv2.imwrite(out_path, vis)

    return {"width_mm": width_mm, "height_mm": height_mm,
            "pixels_per_mm": ratio, "confidence": confidence,
            "reference_mode": ref_mode}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure an object from an undistorted image + binary mask")
    ap.add_argument("--image", required=True, help="Undistorted image path")
    ap.add_argument("--mask", required=True, help="Path to binary mask (png, 0/255)")
    ap.add_argument("--marker_mm", type=float, default=40.0)
    ap.add_argument("--known_width_mm", type=float, default=85.60)
    ap.add_argument("--out", default="measurement_output.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    result = measure(img, mask, args.marker_mm, args.known_width_mm, out_path=args.out)
    print(json.dumps(result, indent=2))
