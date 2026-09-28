# Camera Calibration Report

## Summary

| Parameter | Value |
|---|---|
| Calibration target | Checkerboard — 9×7 inner corners, 25.0 mm squares |
| Images captured | 30 |
| Images used | 30 / 30 |
| Image resolution | 4000 × 3000 px |
| Mean reprojection error | **1.0168 px** |
| Verdict | Above 0.5 px threshold — see analysis below |

---

## Method

Intrinsic camera calibration was performed using OpenCV's `cv2.calibrateCamera()` following the standard checkerboard method:

1. A physical 10×8 square checkerboard (9×7 inner corners, 25 mm square size) was printed and mounted flat.
2. 30 photographs were captured from varied angles, distances, and orientations, keeping the full board in frame and in sharp focus.
3. `cv2.findChessboardCorners()` with `cv2.cornerSubPix()` refinement was used to detect sub-pixel corner positions in each image.
4. The full intrinsic model was solved: focal lengths (fx, fy), principal point (cx, cy), radial distortion (k1, k2, k3), and tangential distortion (p1, p2).
5. Per-image reprojection error was computed as the mean Euclidean distance between projected and detected corner positions.

Script: `calibration/calibrate_camera.py`
Output: `calibration/camera_calibration.npz`

---

## Camera Matrix (K)

```
[[2878.69  0.00    2004.60]
 [0.00     2876.30 1508.50]
 [0.00     0.00    1.00   ]]
```

| Parameter | Value (px) |
|---|---|
| fx (focal length, x) | 2878.69 |
| fy (focal length, y) | 2876.30 |
| cx (principal point, x) | 2004.60 |
| cy (principal point, y) | 1508.50 |

---

## Distortion Coefficients

| Coefficient | Value |
|---|---|
| k1 (radial) | 0.16460 |
| k2 (radial) | −1.43056 |
| p1 (tangential) | 0.000136 |
| p2 (tangential) | 0.000786 |
| k3 (radial) | 2.78984 |

The dominant distortion term is k1 (moderate barrel distortion). k2 and k3 are large in magnitude but partially compensate each other — this is a known behaviour of the 5-parameter OpenCV model when calibrating with a limited angular range of views.

---

## Reprojection Error Analysis

**Mean reprojection error: 1.0168 px**

The target threshold is < 0.5 px (excellent: < 0.3 px). The achieved error is above this target. The primary causes are:

- Mobile-phone sensor with rolling shutter introduces slight per-row timing distortion not modelled by the standard pinhole+polynomial model.
- Several images were captured at near-parallel angles to the board, reducing geometric diversity.
- The higher-order radial terms (k2, k3) are less well constrained at moderate field angles.

Despite the above-target reprojection error, undistortion still meaningfully corrects the dominant barrel distortion and is applied unconditionally before all measurement operations. The measurement pipeline accounts for residual error through the co-planar reference object method (see `docs/MEASUREMENT_REPORT.md`).

### Per-Image Reprojection Error

| Image | Error (px) | Image | Error (px) |
|---|---|---|---|
| IMG_20260927_214947.jpg | 0.7784 | IMG_20260927_215015.jpg | 0.8938 |
| IMG_20260927_214948.jpg | 0.8505 | IMG_20260927_215017.jpg | 1.2197 |
| IMG_20260927_214950.jpg | 0.6897 | IMG_20260927_215018.jpg | 0.9331 |
| IMG_20260927_214953.jpg | 1.1501 | IMG_20260927_215020.jpg | 0.8311 |
| IMG_20260927_214955.jpg | 0.9925 | IMG_20260927_215021.jpg | 1.0084 |
| IMG_20260927_214956.jpg | 1.0454 | IMG_20260927_215022.jpg | 1.0211 |
| IMG_20260927_214958.jpg | 0.9089 | IMG_20260927_215023.jpg | 0.9040 |
| IMG_20260927_214959.jpg | 0.9364 | IMG_20260927_215025.jpg | 1.2147 |
| IMG_20260927_215001.jpg | 1.9597 | IMG_20260927_215026.jpg | 1.6029 |
| IMG_20260927_215003.jpg | 0.7318 | IMG_20260927_215027.jpg | 0.6746 |
| IMG_20260927_215005.jpg | 0.5438 | IMG_20260927_215029.jpg | 0.7229 |
| IMG_20260927_215006.jpg | 0.6807 | IMG_20260927_215030.jpg | 1.0136 |
| IMG_20260927_215008.jpg | 1.5527 | IMG_20260927_215032.jpg | 1.5068 |
| IMG_20260927_215009.jpg | 1.4407 | | |
| IMG_20260927_215011.jpg | 0.9061 | | |
| IMG_20260927_215012.jpg | 0.8964 | | |
| IMG_20260927_215013.jpg | 0.8922 | | |

---

## Undistortion

All object images are undistorted before labelling, training, and measurement using:

```python
cv2.undistort(img, K, dist, None, new_camera_matrix)
```

with `alpha=0` (no black border pixels retained). The optimal new camera matrix is computed via `cv2.getOptimalNewCameraMatrix()` and the principal-point offset is subtracted so downstream pixel coordinates remain consistent.

Script: `calibration/undistort.py`

---

## Assumptions & Limitations

- Calibration was performed with a flat printed board. Any warping of the print introduces systematic error.
- The 5-coefficient distortion model (k1, k2, p1, p2, k3) is standard for moderate wide-angle lenses. A rational polynomial model might reduce error further for this lens.
- Reprojection error reflects the fit to calibration images; generalisation to arbitrary poses is not separately validated here.
- The intrinsic parameters are valid only for this fixed camera configuration (fixed focal length, no digital zoom).
