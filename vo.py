"""
MonocularSLAM: a small, fast, sparse-point-cloud monocular visual
SLAM/odometry pipeline built for interactive (Streamlit) use.

Pipeline
--------
1. Every sampled video frame is fed through pyramidal KLT optical-flow
   tracking (fast; avoids descriptor computation/matching every frame),
   chained continuously frame-to-frame so tracks survive small motions.
2. Forward-backward flow checks discard unreliable tracks (cheap outlier
   rejection before any geometric step).
3. **Adaptive keyframe selection.** Two-view pose estimation is only
   attempted once the *cumulative* median pixel displacement of the
   tracked points since the last keyframe passes a minimum-parallax
   threshold. This is the single biggest lever against monocular drift:
   estimating an essential matrix between frames with too little
   translation relative to rotation (the common case for consecutive
   video frames at 25-60 fps) is a classically ill-conditioned problem --
   `cv2.recoverPose`'s own chirality check will reject nearly every point
   because the decomposed translation direction is essentially noise.
   Waiting for genuine parallax before triangulating turns those
   near-degenerate estimates into well-conditioned ones.
4. 5-point essential-matrix pose estimation with RANSAC
   (`cv2.findEssentialMat` / `cv2.recoverPose`), plus a minimum-inlier-ratio
   sanity check so an unlucky RANSAC draw can't quietly inject a bad pose
   (see `utils.estimate_relative_pose`).
5. Linear triangulation of the surviving inlier correspondences.
6. Relative-scale recovery: the previous keyframe-to-keyframe triangulation
   and this one share tracked points, so the median ratio of pairwise 3D
   distances gives a robust scale factor that keeps translation magnitude
   consistent across keyframes instead of drifting arbitrarily
   (`utils.estimate_relative_scale`).
7. Periodic sliding-window pose-only refinement ("mini bundle adjustment",
   see `ba.py`) jointly re-optimizes the last few keyframe poses against
   already-triangulated map points, damping local error accumulation.
8. Feature replenishment when the tracked set thins out, so tracking
   survives occlusion / features leaving the field of view.

This is not a substitute for full ORB-SLAM-style global bundle adjustment
and loop closure, but keyframe-gated two-view geometry + relative-scale
recovery + windowed pose refinement are well-understood, lightweight
mitigations against the two dominant monocular-VO error sources
(unstable low-parallax pose estimates and unconstrained scale drift),
and they fit a tight real-time compute budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import cv2

from . import utils
from . import ba as ba_mod


@dataclass
class SLAMConfig:
    resize_width: int = 480          # frames are downscaled to this width for speed
    max_features: int = 400          # cap on tracked corners
    min_features: int = 150          # replenish when tracked count drops below this
    frame_stride: int = 2            # process every Nth raw video frame
    hfov_deg: float = 65.0           # assumed horizontal FOV for uncalibrated K
    min_keyframe_parallax_px: float = 8.0   # cumulative median flow that triggers a keyframe
    pose_min_inlier_ratio: float = 0.35     # reject a pose estimate below this inlier fraction
    max_point_depth: float = 1e6            # cheirality / sanity depth cap (unit-baseline units)
    ba_enabled: bool = True
    ba_window: int = 5               # number of trailing keyframes refined jointly
    ba_every: int = 2                # run the mini-BA every N keyframes
    ba_max_nfev: int = 25
    max_points_kept: int = 20000     # safety cap on the accumulated point cloud
    max_keyframes: int = 400         # safety cap so pathological inputs can't run forever


@dataclass
class SLAMResult:
    trajectory: np.ndarray                 # (F, 3) keyframe camera centers, world coords
    point_cloud: np.ndarray                # (M, 3)
    point_colors: np.ndarray               # (M, 3) uint8 RGB, aligned with point_cloud
    n_frames_total: int
    n_frames_processed: int
    n_keyframes: int
    elapsed_sec: float
    fps_processed: float
    frame_size: tuple[int, int]
    K: np.ndarray
    warnings: list[str] = field(default_factory=list)


class MonocularSLAM:
    def __init__(self, config: SLAMConfig | None = None):
        self.cfg = config or SLAMConfig()

    # ------------------------------------------------------------------ #
    def process_video(self, video_path: str, progress_cb=None) -> SLAMResult:
        cfg = self.cfg
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = cfg.resize_width / max(src_w, 1)
        W, H = int(src_w * scale), int(src_h * scale)
        W, H = max(W, 32), max(H, 32)

        K = utils.intrinsics_from_fov(W, H, cfg.hfov_deg)

        t0 = time.perf_counter()
        warnings: list[str] = []

        # --- continuous KLT chain state -------------------------------- #
        prev_gray = None
        cur_pts = np.empty((0, 2), dtype=np.float32)     # tracked points, live positions
        kf_pts = np.empty((0, 2), dtype=np.float32)      # positions of the SAME points at last keyframe
        kf_pts3d = np.empty((0, 3), dtype=np.float64)    # last successful triangulation, aligned to kf_pts
        has3d = np.empty((0,), dtype=bool)

        # --- absolute pose chain (world<-camera) at each keyframe ------- #
        R_wc = np.eye(3)
        t_wc = np.zeros(3)
        trajectory = [(-R_wc.T @ t_wc).copy()]
        pose_history = [(R_wc.copy(), t_wc.copy())]

        all_points_world: list[np.ndarray] = []
        all_points_color: list[np.ndarray] = []

        n_keyframes = 0
        processed = 0
        raw_frame_idx = -1
        last_frame_small = None

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            raw_frame_idx += 1
            if raw_frame_idx % cfg.frame_stride != 0:
                continue

            frame_small = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            last_frame_small = frame_small
            processed += 1

            if prev_gray is None:
                cur_pts = utils.detect_features(gray, cfg.max_features)
                kf_pts = cur_pts.copy()
                kf_pts3d = np.full((cur_pts.shape[0], 3), np.nan)
                has3d = np.zeros((cur_pts.shape[0],), dtype=bool)
                prev_gray = gray
                if progress_cb:
                    progress_cb(min(raw_frame_idx / max(n_frames_total, 1), 0.999))
                continue

            # ---- 1-2. continuous KLT tracking + FB filtering ----------- #
            p_prev, p_cur, keep = utils.track_features(prev_gray, gray, cur_pts)
            kf_pts = kf_pts[keep]
            kf_pts3d = kf_pts3d[keep]
            has3d = has3d[keep]
            cur_pts = p_cur
            prev_gray = gray

            if cur_pts.shape[0] < 8:
                warnings.append(f"frame {raw_frame_idx}: lost nearly all tracks, re-detecting")
            else:
                disp = np.linalg.norm(cur_pts - kf_pts, axis=1)
                median_parallax = float(np.median(disp))

                if median_parallax >= cfg.min_keyframe_parallax_px and n_keyframes < cfg.max_keyframes:
                    R_rel, t_rel_unit, inliers = utils.estimate_relative_pose(
                        kf_pts, cur_pts, K, min_inlier_ratio=cfg.pose_min_inlier_ratio
                    )
                    if R_rel is not None:
                        p_kf_in = kf_pts[inliers]
                        p_cur_in = cur_pts[inliers]
                        pts3d_prev_kf = kf_pts3d[inliers]
                        has3d_in = has3d[inliers]

                        # ---- 5. triangulate at unit baseline ------------ #
                        local_unit = utils.triangulate(
                            K, np.eye(3), np.zeros(3), R_rel, t_rel_unit, p_kf_in, p_cur_in
                        )
                        cmask = utils.cheirality_mask(local_unit, R_rel, t_rel_unit, cfg.max_point_depth)

                        # ---- 6. relative scale recovery ----------------- #
                        common = cmask & has3d_in
                        if common.sum() >= 6:
                            scale = utils.estimate_relative_scale(
                                pts3d_prev_kf[common], local_unit[common]
                            )
                            scale = float(np.clip(scale, 1e-3, 1e3))
                        else:
                            scale = 1.0

                        t_rel = t_rel_unit * scale
                        local_scaled = local_unit * scale

                        # world-space points: local pts are expressed in the
                        # *previous keyframe's* camera-local frame (P1=[I|0])
                        pts_world = (R_wc.T @ (local_scaled[cmask] - t_wc).T).T
                        px = p_cur_in[cmask]
                        cols_bgr = frame_small[
                            np.clip(px[:, 1].astype(int), 0, H - 1),
                            np.clip(px[:, 0].astype(int), 0, W - 1),
                        ]
                        if pts_world.shape[0] > 0:
                            all_points_world.append(pts_world)
                            all_points_color.append(cols_bgr[:, ::-1])  # BGR -> RGB

                        # ---- update absolute pose chain ----------------- #
                        R_wc = R_rel @ R_wc
                        t_wc = R_rel @ t_wc + t_rel
                        n_keyframes += 1
                        trajectory.append((-R_wc.T @ t_wc).copy())
                        pose_history.append((R_wc.copy(), t_wc.copy()))

                        # ---- 7. periodic windowed pose refinement ------- #
                        if cfg.ba_enabled and (n_keyframes % cfg.ba_every == 0) and len(pose_history) >= 2:
                            win = min(cfg.ba_window, len(pose_history))
                            window_poses = pose_history[-win:]
                            obs = _build_window_observations(
                                window_poses, all_points_world, all_points_color, K, max_obs=400
                            )
                            if len(obs) >= 8:
                                refined = ba_mod.refine_window(K, window_poses, obs, max_nfev=cfg.ba_max_nfev)
                                pose_history[-win:] = refined
                                R_wc, t_wc = refined[-1]
                                trajectory[-win:] = [(-R.T @ t) for R, t in refined]

                        # this frame becomes the new keyframe reference.
                        # kf_pts3d must align with the FULL kf_pts (all
                        # currently tracked points), not just the
                        # RANSAC-inlier subset used above, so rebuild it
                        # explicitly over the full set.
                        kf_pts = cur_pts.copy()
                        full_pts3d = np.full((cur_pts.shape[0], 3), np.nan)
                        full_has3d = np.zeros((cur_pts.shape[0],), dtype=bool)
                        inlier_idx = np.where(inliers)[0]
                        full_pts3d[inlier_idx] = local_unit
                        full_has3d[inlier_idx] = cmask
                        kf_pts3d = full_pts3d
                        has3d = full_has3d
                    # if pose estimation failed (low inliers / degenerate),
                    # we deliberately do NOT reset the keyframe reference:
                    # tracking keeps accumulating parallax and we retry on
                    # the next processed frame, which is exactly the
                    # behaviour that recovers from a single bad frame pair
                    # without throwing away a whole keyframe's worth of
                    # accumulated baseline.

            # ---- 8. feature replenishment ---------------------------------- #
            if cur_pts.shape[0] < cfg.min_features:
                mask = np.full((H, W), 255, dtype=np.uint8)
                for x, y in cur_pts.astype(int):
                    cv2.circle(mask, (int(x), int(y)), 6, 0, -1)
                n_new = cfg.max_features - cur_pts.shape[0]
                new_pts = utils.detect_features(gray, max(n_new, 0), mask=mask)
                if new_pts.shape[0] > 0:
                    cur_pts = np.vstack([cur_pts, new_pts])
                    kf_pts = np.vstack([kf_pts, new_pts])
                    kf_pts3d = np.vstack([kf_pts3d, np.full((new_pts.shape[0], 3), np.nan)])
                    has3d = np.concatenate([has3d, np.zeros((new_pts.shape[0],), dtype=bool)])

            if progress_cb:
                progress_cb(min(raw_frame_idx / max(n_frames_total, 1), 0.999))

            if sum(p.shape[0] for p in all_points_world) > cfg.max_points_kept:
                break

        cap.release()
        elapsed = time.perf_counter() - t0

        point_cloud = (np.concatenate(all_points_world, axis=0)
                       if all_points_world else np.empty((0, 3)))
        point_colors = (np.concatenate(all_points_color, axis=0).astype(np.uint8)
                         if all_points_color else np.empty((0, 3), dtype=np.uint8))

        if point_cloud.shape[0] > cfg.max_points_kept:
            sel = np.random.choice(point_cloud.shape[0], cfg.max_points_kept, replace=False)
            point_cloud = point_cloud[sel]
            point_colors = point_colors[sel]

        if n_keyframes == 0:
            warnings.append(
                "No keyframe passed the minimum-parallax / pose-quality thresholds -- "
                "try a video with more camera translation, or lower "
                "'minimum keyframe parallax' in the sidebar."
            )

        return SLAMResult(
            trajectory=np.array(trajectory),
            point_cloud=point_cloud,
            point_colors=point_colors,
            n_frames_total=n_frames_total,
            n_frames_processed=processed,
            n_keyframes=n_keyframes,
            elapsed_sec=elapsed,
            fps_processed=(processed / elapsed) if elapsed > 0 else 0.0,
            frame_size=(W, H),
            K=K,
            warnings=warnings,
        )


def _build_window_observations(window_poses, all_points_world, all_points_color, K, max_obs=400):
    """
    Build a small set of (pose_idx_in_window, xyz, uv) observations for the
    mini-BA by reprojecting recently added world points into each pose in
    the window and keeping the ones that land in front of the camera. This
    keeps the refinement anchored to real map points without needing a full
    per-point visibility graph.
    """
    if not all_points_world:
        return []
    recent = np.concatenate(all_points_world[-3:], axis=0)
    if recent.shape[0] == 0:
        return []
    if recent.shape[0] > max_obs:
        idx = np.random.choice(recent.shape[0], max_obs, replace=False)
        recent = recent[idx]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    obs = []
    for p_idx, (R, t) in enumerate(window_poses):
        pts_cam = (R @ recent.T).T + t.reshape(1, 3)
        z = pts_cam[:, 2]
        valid = z > 1e-3
        u = fx * pts_cam[valid, 0] / z[valid] + cx
        v = fy * pts_cam[valid, 1] / z[valid] + cy
        for uu, vv, xyz in zip(u, v, recent[valid]):
            obs.append((p_idx, xyz, np.array([uu, vv])))
    return obs
