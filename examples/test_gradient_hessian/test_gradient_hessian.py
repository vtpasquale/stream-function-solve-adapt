
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gradient_hessian.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from stream_function_solve_adapt.TriMesh import TriMesh
from stream_function_solve_adapt.TriMesh_assembly import (
    triangle_geometry, compute_gradients, compute_hessian,
)


def phi_analytic(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def grad_analytic(x, y):
    dphidx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
    dphidy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
    return dphidx, dphidy


def hessian_analytic(x, y):
    Hxx = -(np.pi ** 2) * np.sin(np.pi * x) * np.cos(np.pi * y)
    Hyy = -(np.pi ** 2) * np.sin(np.pi * x) * np.cos(np.pi * y)
    Hxy = -(np.pi ** 2) * np.cos(np.pi * x) * np.sin(np.pi * y)
    return Hxx, Hyy, Hxy


def generate_parallelogram_mesh(n, angle_deg):
    theta = np.deg2rad(angle_deg)
    xi = np.linspace(0.0, 1.0, n)
    eta = np.linspace(0.0, 1.0, n)
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")

    X = XI + ETA * np.cos(theta)
    Y = ETA * np.sin(theta)
    nodes = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])

    def node_id(i, j):
        return i * n + j

    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            n00, n10 = node_id(i, j), node_id(i + 1, j)
            n01, n11 = node_id(i, j + 1), node_id(i + 1, j + 1)
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])

    return TriMesh(nodes, np.array(triangles, dtype=int))


def _tricontour_row(fig, axs, triang, analytic, recovered, label):
    error = recovered - analytic
    vmin, vmax = analytic.min(), analytic.max()

    c0 = axs[0].tricontourf(triang, analytic, levels=20, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"{label}: analytic")
    fig.colorbar(c0, ax=axs[0])

    c1 = axs[1].tricontourf(triang, recovered, levels=20, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"{label}: recovered")
    fig.colorbar(c1, ax=axs[1])

    lim = np.max(np.abs(error)) + 1e-14
    c2 = axs[2].tricontourf(triang, error, levels=20, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].set_title(f"{label}: error")
    fig.colorbar(c2, ax=axs[2])

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    rms = np.sqrt(np.mean(error ** 2))
    max_err = np.max(np.abs(error))
    return rms, max_err


def evaluate_angle(n, angle_deg, out_dir):
    mesh = generate_parallelogram_mesh(n, angle_deg)
    area, _, _ = triangle_geometry(mesh)
    if np.any(area <= 0):
        raise ValueError(f"Non-positive element area at angle={angle_deg} deg.")

    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    phi = phi_analytic(x, y)

    gx_a, gy_a = grad_analytic(x, y)
    Hxx_a, Hyy_a, Hxy_a = hessian_analytic(x, y)

    grad_num = compute_gradients(mesh, phi)
    hess_num = compute_hessian(mesh, phi)

    triang = mtri.Triangulation(x, y, mesh.triangles)

    fig, axs = plt.subplots(5, 3, figsize=(12, 18))
    metrics = {}
    rows = [
        ("dphi/dx", gx_a, grad_num[:, 0]),
        ("dphi/dy", gy_a, grad_num[:, 1]),
        ("Hxx", Hxx_a, hess_num[:, 0]),
        ("Hyy", Hyy_a, hess_num[:, 1]),
        ("Hxy", Hxy_a, hess_num[:, 2]),
    ]
    for row_axs, (label, analytic, recovered) in zip(axs, rows):
        rms, max_err = _tricontour_row(fig, row_axs, triang, analytic, recovered, label)
        metrics[label] = (rms, max_err)

    fig.suptitle(f"Gradient / Hessian recovery -- parallelogram angle = {angle_deg} deg", y=1.0)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"recovery_angle_{angle_deg:03d}deg.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return metrics


def plot_error_summary(angles, all_metrics, out_dir):
    labels = list(all_metrics[angles[0]].keys())
    fig, (ax_rms, ax_max) = plt.subplots(1, 2, figsize=(11, 4.5))

    for label in labels:
        rms_vals = [all_metrics[a][label][0] for a in angles]
        max_vals = [all_metrics[a][label][1] for a in angles]
        ax_rms.plot(angles, rms_vals, marker="o", label=label)
        ax_max.plot(angles, max_vals, marker="o", label=label)

    for ax, title in [(ax_rms, "RMS error"), (ax_max, "Max error")]:
        ax.set_xlabel("parallelogram angle (deg); 90 = undistorted square")
        ax.set_ylabel(title)
        ax.set_title(title + " vs. mesh distortion")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_vs_distortion_summary.png"), dpi=110)
    plt.close(fig)


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    n = 26
    angles = [90, 75, 60, 45, 30, 15]

    all_metrics = {}
    for angle_deg in angles:
        print(f"Evaluating angle = {angle_deg} deg ...")
        all_metrics[angle_deg] = evaluate_angle(n, angle_deg, out_dir)

    plot_error_summary(angles, all_metrics, out_dir)

    print("\\nRMS error by component and distortion angle:")
    header = "angle".rjust(8) + "".join(l.rjust(14) for l in all_metrics[angles[0]].keys())
    print(header)
    for angle_deg in angles:
        row = f"{angle_deg:8d}" + "".join(
            f"{all_metrics[angle_deg][l][0]:14.4e}" for l in all_metrics[angle_deg]
        )
        print(row)

    print(f"\\nPer-angle contour plots and the summary plot were written to ./{out_dir}/")


if __name__ == "__main__":
    main()
