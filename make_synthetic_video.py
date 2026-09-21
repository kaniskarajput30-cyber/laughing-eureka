"""
Generates a synthetic test video: a textured 3D point field rendered from a
camera moving on a smooth path with known ground-truth trajectory. Useful
for sanity-checking the SLAM pipeline (does the recovered trajectory shape
resemble the ground truth?) and for timing benchmarks, without needing a
real-world video file.
"""
import numpy as np
import cv2


def make_video(path: str, seconds: float = 10.0, fps: float = 30.0,
                width: int = 640, height: int = 360, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_frames = int(seconds * fps)

    # random textured 3D points in front of the camera path
    n_pts = 900
    pts_world = rng.uniform(low=[-6, -4, 8], high=[6, 4, 30], size=(n_pts, 3))
    colors = rng.integers(40, 255, size=(n_pts, 3))

    fx = fy = width * 0.9
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    ground_truth = []
    for i in range(n_frames):
        t = i / n_frames
        # smooth lateral + slight forward camera motion (a gentle arc)
        cam_center = np.array([4.0 * np.sin(2 * np.pi * t * 0.5), 0.3 * np.sin(2 * np.pi * t * 1.0), 0.0])
        yaw = 0.35 * np.sin(2 * np.pi * t * 0.5)
        Rwc = cv2.Rodrigues(np.array([0, yaw, 0]))[0]  # world->cam rotation
        twc = -Rwc @ cam_center
        ground_truth.append(cam_center)

        frame = np.full((height, width, 3), 20, dtype=np.uint8)
        pts_cam = (Rwc @ pts_world.T).T + twc
        z = pts_cam[:, 2]
        valid = z > 0.5
        u = fx * pts_cam[valid, 0] / z[valid] + cx
        v = fy * pts_cam[valid, 1] / z[valid] + cy
        cols = colors[valid]
        for (uu, vv), col in zip(zip(u, v), cols):
            if 0 <= uu < width and 0 <= vv < height:
                cv2.circle(frame, (int(uu), int(vv)), 3, tuple(int(c) for c in col), -1)
        writer.write(frame)

    writer.release()
    return np.array(ground_truth), K


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_10s.mp4"
    gt, K = make_video(out)
    np.save(out + ".gt.npy", gt)
    print("wrote", out, "frames:", gt.shape)
