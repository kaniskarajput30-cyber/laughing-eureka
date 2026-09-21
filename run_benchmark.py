import sys, time
sys.path.insert(0, ".")
import numpy as np
from slam.vo import MonocularSLAM, SLAMConfig

video = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_10s.mp4"

cfg = SLAMConfig()
slam = MonocularSLAM(cfg)

t0 = time.perf_counter()
result = slam.process_video(video)
wall = time.perf_counter() - t0

print(f"video: {video}")
print(f"frames total (source): {result.n_frames_total}")
print(f"frames processed:      {result.n_frames_processed}")
print(f"internal elapsed_sec:  {result.elapsed_sec:.3f}")
print(f"wall elapsed_sec:      {wall:.3f}")
print(f"processed fps:         {result.fps_processed:.1f}")
print(f"frame size:            {result.frame_size}")
print(f"trajectory points:     {result.trajectory.shape}")
print(f"point cloud size:      {result.point_cloud.shape}")
print(f"warnings ({len(result.warnings)}):", result.warnings[:5])

traj = result.trajectory
path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
print(f"total estimated path length (arbitrary scale units): {path_len:.3f}")

gt_path = video + ".gt.npy"
try:
    gt = np.load(gt_path)
    gt_len = np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))
    print(f"ground-truth path length (metric units): {gt_len:.3f}")
    print(f"implied scale factor (gt/est): {gt_len/path_len:.4f}")
except FileNotFoundError:
    pass

assert wall <= 10.0, f"FAILED perf target: {wall:.2f}s > 10s"
print("PERF TARGET MET (<=10s for a 10s video)")
