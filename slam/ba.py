"""
Lightweight sliding-window pose refinement ("mini bundle adjustment").

Full bundle adjustment (jointly optimizing every pose and every map point
over the whole sequence) gives the best accuracy but is far too slow for a
"10s video -> <=10s processing" budget. Instead we run a cheap windowed
*pose-only* refinement: every few frames we re-optimize the last W camera
poses so that their reprojection of the already-triangulated (and therefore
fixed) map points is jointly consistent, using Levenberg-Marquardt via
scipy. Points are treated as fixed anchors, so the problem is small
(6 parameters per pose in the window) and runs in milliseconds even for
a handful of keyframes with a few hundred point observations -- enough to
noticeably damp local accumulation of pose-estimation error without paying
for full structure-from-motion optimization.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import least_squares


def _pose_to_vec(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t.reshape(3)])


def _vec_to_pose(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(v[:3])
    t = v[3:6]
    return R, t


def _residuals(params: np.ndarray, n_poses: int, K: np.ndarray,
               obs_pose_idx: np.ndarray, obs_point_xyz: np.ndarray,
               obs_uv: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    res = np.empty(obs_uv.shape[0] * 2, dtype=np.float64)

    poses = params.reshape(n_poses, 6)
    for p in range(n_poses):
        sel = obs_pose_idx == p
        if not np.any(sel):
            continue
        R, t = _vec_to_pose(poses[p])
        pts_cam = (R @ obs_point_xyz[sel].T).T + t.reshape(1, 3)
        z = np.clip(pts_cam[:, 2], 1e-6, None)
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy
        proj = np.stack([u, v], axis=1)
        err = (proj - obs_uv[sel]).reshape(-1)
        idxs = np.where(sel)[0]
        for k, i in enumerate(idxs):
            res[2 * i:2 * i + 2] = err[2 * k:2 * k + 2]
    return res


def refine_window(K: np.ndarray, window_poses: list[tuple[np.ndarray, np.ndarray]],
                   observations: list[tuple[int, np.ndarray, np.ndarray]],
                   max_nfev: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    window_poses: list of (R, t) for the poses in the sliding window (world<-cam).
    observations: list of (pose_idx, point_xyz, uv) -- pose_idx indexes window_poses,
                  point_xyz is the fixed triangulated 3D map point, uv is its pixel
                  observation in that pose's frame.
    Returns the refined list of (R, t), same length/order as window_poses.
    """
    n_poses = len(window_poses)
    if n_poses < 2 or len(observations) < 8:
        return window_poses

    x0 = np.concatenate([_pose_to_vec(R, t) for R, t in window_poses])
    obs_pose_idx = np.array([o[0] for o in observations], dtype=np.int64)
    obs_point_xyz = np.array([o[1] for o in observations], dtype=np.float64)
    obs_uv = np.array([o[2] for o in observations], dtype=np.float64)

    # anchor the first pose in the window so the whole window can't drift
    # off to a degenerate solution; only poses 1..n-1 are truly free.
    def wrapped_residuals(params):
        full = x0.copy()
        full[6:] = params  # first pose fixed
        return _residuals(full, n_poses, K, obs_pose_idx, obs_point_xyz, obs_uv)

    if n_poses == 1:
        return window_poses

    free0 = x0[6:]
    try:
        result = least_squares(
            wrapped_residuals, free0, method="lm", max_nfev=max_nfev * max(1, n_poses - 1)
        )
    except Exception:
        return window_poses

    full = x0.copy()
    full[6:] = result.x
    refined = []
    for p in range(n_poses):
        R, t = _vec_to_pose(full[p * 6:p * 6 + 6])
        refined.append((R, t))
    return refined
