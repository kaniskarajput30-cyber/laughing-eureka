"""
Low-level computer-vision helpers for the monocular VO/SLAM pipeline.

All functions operate on plain numpy arrays / OpenCV objects so they can be
unit-tested independently of Streamlit or the higher-level SLAM state
machine in `vo.py`.
"""
from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Camera intrinsics
# --------------------------------------------------------------------------- #
def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """
    Build a plausible pinhole camera matrix K when the real calibration is
    unknown (the common case for an arbitrary uploaded video). The focal
    length is derived from an assumed horizontal field of view, which is a
    standard approximation for uncalibrated monocular VO demos.
    """
    hfov = np.deg2rad(hfov_deg)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    fy = fx  # square pixels assumption
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Feature detection / tracking
# --------------------------------------------------------------------------- #
def detect_features(gray: np.ndarray, max_features: int, quality: float = 0.01,
                     min_distance: int = 8, mask: np.ndarray | None = None) -> np.ndarray:
    """Shi-Tomasi corners -- fast to detect and well suited to KLT tracking."""
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_features, qualityLevel=quality,
        minDistance=min_distance, mask=mask, blockSize=7,
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def track_features(prev_gray: np.ndarray, cur_gray: np.ndarray,
                    prev_pts: np.ndarray, win_size: int = 21,
                    max_level: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pyramidal Lucas-Kanade optical-flow tracking between two frames.
    Forward-backward error is used to reject unreliable tracks, which is a
    cheap and effective outlier filter before the geometric RANSAC stage.
    Returns (prev_pts_ok, cur_pts_ok, keep_mask_into_original_indices).
    """
    if prev_pts.shape[0] == 0:
        return prev_pts, prev_pts, np.zeros((0,), dtype=bool)

    lk_params = dict(winSize=(win_size, win_size), maxLevel=max_level,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    cur_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_pts, None, **lk_params)
    back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, cur_pts, None, **lk_params)

    status_f = status_f.reshape(-1).astype(bool)
    status_b = status_b.reshape(-1).astype(bool)
    fb_err = np.linalg.norm(prev_pts - back_pts, axis=1)

    keep = status_f & status_b & (fb_err < 1.5)
    return prev_pts[keep], cur_pts[keep], keep


# --------------------------------------------------------------------------- #
# Relative pose estimation
# --------------------------------------------------------------------------- #
def estimate_relative_pose(pts_prev: np.ndarray, pts_cur: np.ndarray, K: np.ndarray,
                            min_inlier_ratio: float = 0.25):
    """
    Robust 5-point essential-matrix estimation + pose recovery.
    Returns (R, t, inlier_mask) with t a unit vector (monocular scale is
    unknown from two views alone -- see `estimate_relative_scale` below for
    how the pipeline fixes an absolute scale to curb drift).

    RANSAC's random sampling occasionally converges to a degenerate
    low-consensus solution (a handful of "inliers" that happen to fit some
    spurious essential matrix) even when the vast majority of tracked
    points are perfectly good correspondences. Blindly trusting that result
    would collapse the tracked point set for no real reason and cascade
    into failures on the next frame, so any solution whose inlier ratio is
    implausibly low is rejected as a failed pose estimate rather than acted
    on -- the caller then keeps the full previous track set and simply
    retries on the next frame.
    """
    n = pts_prev.shape[0]
    if n < 8:
        return None, None, None

    E, mask = cv2.findEssentialMat(
        pts_prev, pts_cur, K, method=cv2.RANSAC, prob=0.999, threshold=1.5
    )
    if E is None or E.shape != (3, 3):
        return None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts_prev, pts_cur, K, mask=mask)
    inliers = (mask_pose.reshape(-1) > 0)

    min_needed = max(8, int(min_inlier_ratio * n))
    if inliers.sum() < min_needed:
        return None, None, None

    return R, t.reshape(3), inliers


# --------------------------------------------------------------------------- #
# Triangulation
# --------------------------------------------------------------------------- #
def triangulate(K: np.ndarray, R1: np.ndarray, t1: np.ndarray,
                 R2: np.ndarray, t2: np.ndarray,
                 pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of matched 2D points from two known poses."""
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def cheirality_mask(pts3d: np.ndarray, R: np.ndarray, t: np.ndarray, max_depth: float = 200.0) -> np.ndarray:
    """Keep only points in front of both cameras and within a sane depth range."""
    depth1 = pts3d[:, 2]
    pts_cam2 = (R @ pts3d.T).T + t.reshape(1, 3)
    depth2 = pts_cam2[:, 2]
    return (depth1 > 0) & (depth2 > 0) & (depth1 < max_depth) & (depth2 < max_depth)


# --------------------------------------------------------------------------- #
# Drift mitigation: relative scale recovery
# --------------------------------------------------------------------------- #
def estimate_relative_scale(prev_pts3d: np.ndarray, cur_pts3d: np.ndarray,
                             sample: int = 200) -> float:
    """
    Classic monocular-VO trick to fix scale drift frame-to-frame: pose and
    triangulation from a single image pair only recover translation up to
    an unknown scale, and naively re-scaling by a constant lets absolute
    scale wander over time. Instead we look at points triangulated in BOTH
    the previous and the current relative-pose windows (same track ids) and
    compare pairwise 3D distances -- the ratio of corresponding distances
    is a robust (median-based) estimate of how much bigger/smaller the new
    triangulation is, which lets us rescale the new translation into the
    previous, already-consistent, coordinate scale.
    """
    n = min(len(prev_pts3d), len(cur_pts3d))
    if n < 6:
        return 1.0

    idx = np.random.choice(n, size=min(sample, n), replace=False)
    a = prev_pts3d[idx]
    b = cur_pts3d[idx]

    # pairwise distances within each small random sample
    da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    db = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)

    iu = np.triu_indices(len(idx), k=1)
    da, db = da[iu], db[iu]
    valid = db > 1e-6
    if valid.sum() < 5:
        return 1.0

    ratios = da[valid] / db[valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    return float(np.median(ratios))
