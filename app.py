"""
Monocular RGB Sparse-Point-Cloud SLAM -- Streamlit front end.

Upload a single-camera RGB video; the app estimates the camera trajectory
and a sparse 3D point cloud from 2D feature tracking + two-view geometry,
using keyframe-gated pose estimation, relative-scale recovery and a
sliding-window pose refinement to curb monocular drift (see slam/vo.py for
the algorithm write-up). Tuned to process a ~10 second clip in well under
10 seconds on a typical CPU-only machine.
"""
import os
import tempfile
import time

import numpy as np
import streamlit as st

from slam.vo import MonocularSLAM, SLAMConfig
from slam.viz import make_scene_figure, export_ply

st.set_page_config(page_title="Monocular RGB SLAM", layout="wide", page_icon="\U0001F4F7")

st.title("\U0001F4F7 Monocular RGB Sparse-Point-Cloud SLAM")
st.caption(
    "Upload a short single-camera video. The app tracks 2D features, recovers "
    "relative camera poses with keyframe-gated two-view geometry, mitigates "
    "monocular scale drift, and reconstructs a sparse 3D point cloud + camera "
    "trajectory -- all in the browser, no calibration file required."
)

# --------------------------------------------------------------------- #
# Sidebar: parameters
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")

    st.subheader("Performance")
    resize_width = st.slider("Processing width (px)", 240, 960, 480, step=40,
                              help="Frames are downscaled to this width before tracking. "
                                   "Lower = faster, less accurate.")
    frame_stride = st.slider("Frame stride", 1, 6, 2,
                              help="Process every Nth raw video frame. Higher = faster.")
    max_features = st.slider("Max tracked features", 100, 800, 400, step=50)

    st.subheader("Camera model")
    hfov_deg = st.slider("Assumed horizontal FOV (deg)", 30.0, 110.0, 65.0,
                          help="No calibration file is required, so the focal length is "
                               "derived from an assumed field of view. If you know your "
                               "camera's real FOV, set it here for a more accurate "
                               "reconstruction.")

    st.subheader("Drift mitigation")
    min_parallax = st.slider("Minimum keyframe parallax (px)", 2.0, 25.0, 8.0,
                              help="A new keyframe (and a new pose estimate) is only "
                                   "triggered once tracked points have moved this many "
                                   "pixels on average since the last keyframe. This avoids "
                                   "the classic monocular failure mode of estimating pose "
                                   "from near-zero-baseline frame pairs.")
    ba_enabled = st.checkbox("Enable sliding-window pose refinement (mini bundle adjustment)",
                              value=True)
    ba_window = st.slider("BA window size (keyframes)", 2, 10, 5, disabled=not ba_enabled)

    st.subheader("Output")
    point_size = st.slider("Point size in 3D view", 1.0, 6.0, 2.0)

st.divider()

uploaded = st.file_uploader("Upload a video (mp4, mov, avi, mkv)",
                             type=["mp4", "mov", "avi", "mkv", "m4v"])

col_left, col_right = st.columns([1, 1])

if uploaded is not None:
    with col_left:
        st.video(uploaded)

    run = st.button("\U0001F680 Run SLAM", type="primary", use_container_width=True)

    if run:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name

        cfg = SLAMConfig(
            resize_width=resize_width,
            max_features=max_features,
            frame_stride=frame_stride,
            hfov_deg=hfov_deg,
            min_keyframe_parallax_px=min_parallax,
            ba_enabled=ba_enabled,
            ba_window=ba_window,
        )
        slam = MonocularSLAM(cfg)

        progress_bar = st.progress(0.0, text="Processing video...")

        def _cb(frac):
            progress_bar.progress(frac, text=f"Processing video... {int(frac * 100)}%")

        t0 = time.perf_counter()
        try:
            result = slam.process_video(video_path, progress_cb=_cb)
        finally:
            os.unlink(video_path)
        wall = time.perf_counter() - t0

        progress_bar.progress(1.0, text="Done.")
        st.session_state["slam_result"] = result
        st.session_state["slam_wall"] = wall

# --------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------- #
if "slam_result" in st.session_state:
    result = st.session_state["slam_result"]
    wall = st.session_state["slam_wall"]

    for w in result.warnings:
        st.warning(w)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Processing time", f"{wall:.2f} s")
    m2.metric("Frames processed", f"{result.n_frames_processed} / {result.n_frames_total}")
    m3.metric("Keyframes", result.n_keyframes)
    m4.metric("Map points", result.point_cloud.shape[0])
    m5.metric("Processing FPS", f"{result.fps_processed:.1f}")

    if wall <= 10.0:
        st.success(f"Processed in {wall:.2f}s -- within the 10s target for a 10s clip.")
    else:
        st.info(
            f"Processed in {wall:.2f}s. For long or high-resolution videos, reduce the "
            f"processing width or increase the frame stride in the sidebar to stay within "
            f"a 10s budget for a 10s clip."
        )

    st.subheader("Sparse 3D map + camera trajectory")
    if result.point_cloud.shape[0] == 0 and result.n_keyframes == 0:
        st.error(
            "No reliable camera motion could be estimated from this video. Monocular "
            "SLAM needs some camera *translation* (sidestepping/moving forward), not just "
            "rotation in place -- try a video with more lateral or forward camera movement, "
            "or lower 'Minimum keyframe parallax' in the sidebar."
        )
    else:
        fig = make_scene_figure(result.point_cloud, result.point_colors, result.trajectory,
                                 point_size=point_size)
        st.plotly_chart(fig, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            ply_path = os.path.join(tempfile.gettempdir(), "sparse_map.ply")
            export_ply(result.point_cloud, result.point_colors, ply_path)
            with open(ply_path, "rb") as f:
                st.download_button("Download point cloud (.ply)", f, file_name="sparse_map.ply",
                                    use_container_width=True)
        with dl2:
            traj_csv = "x,y,z\n" + "\n".join(f"{x},{y},{z}" for x, y, z in result.trajectory)
            st.download_button("Download trajectory (.csv)", traj_csv.encode(),
                                file_name="trajectory.csv", use_container_width=True)

    with st.expander("How this works / limitations"):
        st.markdown(
            """
**Pipeline:** Shi-Tomasi corners are tracked frame-to-frame with pyramidal
Lucas-Kanade optical flow. A new **keyframe** and camera pose is only
computed once tracked points have accumulated enough parallax
(`min_keyframe_parallax_px`), since two-view pose estimation from
near-zero-baseline frame pairs is numerically unstable -- this is the main
lever against drift. Pose comes from a RANSAC 5-point essential-matrix
solve (`cv2.findEssentialMat` + `cv2.recoverPose`), with a minimum-inlier
sanity check so a single unlucky RANSAC draw can't inject a bad pose.
Points are triangulated per keyframe pair, and monocular **scale drift**
is curbed by comparing pairwise 3D distances of points common to
consecutive keyframe triangulations (a robust, median-based relative-scale
estimate). Every few keyframes, a lightweight **sliding-window pose
refinement** ("mini bundle adjustment") jointly re-optimizes the last few
poses against already-triangulated map points.

**Known limitations** (by design, to keep this fast and dependency-light):
- No calibration file is used, so absolute *metric* scale is not
  recoverable -- only self-consistent *relative* scale (a fundamental
  monocular limitation, not specific to this tool).
- No loop closure or global bundle adjustment, so drift over very long or
  looping sequences will still accumulate.
- Map points are not fused/deduplicated across keyframes, so the same
  physical point may appear more than once in the cloud.
- Needs real camera *translation* to work -- pure in-place rotation cannot
  be triangulated by any monocular two-view method.
            """
        )
else:
    st.info("Upload a video and click **Run SLAM** to get started.")
