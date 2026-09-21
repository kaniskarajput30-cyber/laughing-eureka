# laughing-eureka
# Monocular RGB Sparse-Point-Cloud SLAM (Streamlit)

A single-camera (monocular) visual SLAM / odometry tool: upload an RGB
video, and it estimates the camera trajectory and a sparse 3D point cloud
using classic 2D-feature-tracking + two-view geometry, with three
lightweight, well-understood mitigations against monocular drift baked in.
Built to comfortably process a ~10 second clip in under 10 seconds on a
CPU-only machine.

**Live demo:** deploy in a few clicks on [Streamlit Community
Cloud](https://streamlit.io/cloud) (see below).

## What it does

1. **Upload a video** (mp4/mov/avi/mkv) through the browser.
2. **Estimates camera pose and trajectory** frame by frame, using
   pyramidal Lucas-Kanade optical-flow tracking + RANSAC 5-point
   essential-matrix pose recovery (`cv2.findEssentialMat` /
   `cv2.recoverPose`).
3. **Minimizes accumulated pose error and drift** via three techniques
   working together (details in [`slam/vo.py`](slam/vo.py)):
   - **Adaptive keyframe selection** -- a pose is only estimated once
     tracked points have accumulated enough parallax since the last
     keyframe, avoiding the classic monocular failure mode of decomposing
     an essential matrix from a near-zero-baseline (or near-pure-rotation)
     frame pair, which is numerically unstable and a major source of
     spurious drift.
   - **Relative-scale recovery** -- monocular two-view triangulation only
     recovers translation up to an unknown scale; naively chaining
     unit-scale steps lets absolute scale wander. Each keyframe's scale is
     instead fixed by comparing pairwise 3D distances of points common to
     the previous and current triangulation (a robust, median-based
     ratio), keeping translation magnitude self-consistent over time.
   - **Sliding-window pose refinement** ("mini bundle adjustment",
     [`slam/ba.py`](slam/ba.py)) -- every few keyframes, the last several
     poses are jointly re-optimized (Levenberg-Marquardt via SciPy)
     against already-triangulated map points, damping local error
     accumulation without the cost of full global bundle adjustment.
   - A RANSAC minimum-inlier-ratio sanity check also rejects the
     occasional degenerate RANSAC draw before it can inject a bad pose.
4. **Generates and visualizes** the resulting sparse 3D point cloud and
   camera trajectory in an interactive Plotly 3D scene, with `.ply` /
   `.csv` export.
5. **Meets a tight performance budget**: on the reference test machine, a
   10-second 1080p clip processes in **~4-5 seconds** (see
   [Benchmark](#benchmark) below) -- well inside the 10-second target for
   a 10-second video.

## Project layout

```
mono-rgb-slam/
├── app.py                  # Streamlit UI
├── slam/
│   ├── vo.py                # MonocularSLAM pipeline / state machine
│   ├── utils.py              # feature tracking, pose estimation, triangulation, scale recovery
│   ├── ba.py                  # sliding-window pose-only bundle adjustment
│   └── viz.py                  # Plotly 3D visualization + .ply export
├── tests/
│   ├── make_synthetic_video.py  # generates a synthetic video with known ground-truth trajectory
│   └── run_benchmark.py          # runs the pipeline end-to-end and checks the perf target
├── requirements.txt
├── packages.txt              # apt packages needed by opencv on Streamlit Cloud
└── .streamlit/config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL, upload a video, and click **Run SLAM**.

## Deploy via GitHub (Streamlit Community Cloud)

1. Push this project to a new GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Monocular RGB sparse SLAM Streamlit app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Pick your repository/branch and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and the
   apt packages in `packages.txt` (needed for OpenCV) automatically.

The app has no external services or API keys to configure -- everything
runs inside the Streamlit process.

## Benchmark

`tests/run_benchmark.py` generates a synthetic 10-second video with a
*known ground-truth trajectory* (a textured 3D point field seen by a
camera moving on a smooth path) and runs the full pipeline end-to-end,
asserting the 10-second processing budget:

```bash
python tests/make_synthetic_video.py /tmp/synthetic_10s.mp4
python tests/run_benchmark.py /tmp/synthetic_10s.mp4
```

On the reference (CPU-only, containerized) test environment:

| Input                        | Frames | Wall time | Keyframes | Map points |
|-------------------------------|-------:|----------:|----------:|-----------:|
| 480x270 synthetic, 10s @30fps  | 150 processed | ~3.8s | ~14 | ~3900 |
| 1920x1080 synthetic, 10s @30fps| 150 processed | ~4.5s | ~14 | ~3500 |

(Both comfortably under the 10-second target; frames are internally
downscaled to the configurable "processing width", default 480px, which
is what keeps larger source resolutions from materially increasing
runtime.) The recovered trajectory correlates with the synthetic
ground-truth path at r ≈ 0.93-0.97 after the (expected, monocular-scale-
ambiguous) global scale alignment.

For a real-world clip, actual timing depends on video codec, resolution,
scene texture (feature count) and how much genuine camera translation is
present; use the sidebar's **processing width** and **frame stride**
controls to trade accuracy for speed if needed.

## Tuning / parameters (sidebar)

| Setting | Effect |
|---|---|
| Processing width | Frames are downscaled to this width before tracking. Lower = faster, less accurate. |
| Frame stride | Process every Nth raw frame. Higher = faster. |
| Max tracked features | Cap on Shi-Tomasi corners tracked per frame. |
| Assumed horizontal FOV | No calibration file is used; the intrinsic focal length is derived from this. Set it to your camera's real FOV for a more accurate reconstruction. |
| Minimum keyframe parallax | Pixel displacement needed since the last keyframe before a new pose is estimated -- the main drift-control knob. |
| Sliding-window pose refinement | Toggle the mini bundle adjustment; window size controls how many trailing keyframes are jointly refined. |

## Known limitations

This is a compact, dependency-light VO/SLAM pipeline, not a research-grade
system. By design, to keep it fast and simple:

- **No absolute metric scale.** Without a calibration file or an external
  reference (IMU, known object size, stereo baseline), monocular vision
  can only recover *relative*, self-consistent scale -- a fundamental,
  well-known limitation of any single-camera SLAM system, not specific to
  this tool.
- **No loop closure or global bundle adjustment.** Drift is heavily
  mitigated (see above) but will still accumulate over very long or
  loop-heavy sequences; a full SLAM back-end would add pose-graph
  optimization and place recognition.
- **No map-point fusion/deduplication** across keyframes -- the same
  physical surface point can be re-triangulated and appear more than once
  in the cloud.
- **Needs genuine camera translation.** Pure in-place rotation (panning
  without moving) cannot be triangulated by any monocular two-view method;
  the app will surface a warning rather than fabricate a pose in that
  case.

## License

MIT (or your organization's preferred license) -- edit as needed.
