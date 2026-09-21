"""Plotly 3D visualization helpers for the sparse point cloud + trajectory."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def make_scene_figure(point_cloud: np.ndarray, point_colors: np.ndarray,
                       trajectory: np.ndarray, point_size: float = 2.0) -> go.Figure:
    fig = go.Figure()

    if point_cloud.shape[0] > 0:
        colors = (point_colors if point_colors.shape[0] == point_cloud.shape[0]
                  else np.full((point_cloud.shape[0], 3), 160, dtype=np.uint8))
        color_strs = [f"rgb({r},{g},{b})" for r, g, b in colors]
        fig.add_trace(go.Scatter3d(
            x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=color_strs, opacity=0.85),
            name="Sparse map points",
        ))

    if trajectory.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode="lines+markers",
            line=dict(color="crimson", width=5),
            marker=dict(size=3, color="crimson"),
            name="Camera trajectory (keyframes)",
        ))
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]], y=[trajectory[0, 1]], z=[trajectory[0, 2]],
            mode="markers", marker=dict(size=7, color="lime", symbol="diamond"),
            name="Start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]], y=[trajectory[-1, 1]], z=[trajectory[-1, 2]],
            mode="markers", marker=dict(size=7, color="orange", symbol="diamond"),
            name="End",
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=650,
    )
    return fig


def export_ply(point_cloud: np.ndarray, point_colors: np.ndarray, path: str) -> None:
    """Write the sparse point cloud to a colored ASCII .ply file."""
    n = point_cloud.shape[0]
    colors = (point_colors if point_colors.shape[0] == n
              else np.full((n, 3), 160, dtype=np.uint8))
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(point_cloud, colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
