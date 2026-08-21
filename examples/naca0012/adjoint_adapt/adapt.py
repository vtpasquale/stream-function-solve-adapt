# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:31:31 2026

@author: aricciar
"""
import os
import time
import subprocess
import shutil

import pandas as pd
import numpy as np

import stream_function_solve_adapt.TriMesh as tm
import stream_function_solve_adapt.solve as sf
from stream_function_solve_adapt.lift_adjoint_metric import lift_adjoint_metric

import pyLibMeshb.libMeshb as lm


def solve(project, adapt_step):
    start_time = time.perf_counter()
    mesh_file = f"{project}_{adapt_step}.meshb"
    alpha_rad = float(np.deg2rad(10.0))

    # Read mesh and update edge ids
    mesh = tm.TriMesh.from_file(mesh_file)

    # Fix node ordering for positive element areas
    mesh.triangles = mesh.triangles[:, ::-1]

    # Update edge ids
    airfoil_edge_id = 0
    farfield_edge_id = 1
    mesh.edge_ids[mesh.edge_ids == 2] = farfield_edge_id
    mesh.edge_ids[mesh.edge_ids == 3] = airfoil_edge_id
    mesh.edge_ids[mesh.edge_ids == 4] = airfoil_edge_id

    # Reverse edge order at airfoil so circulation integral is counterclockwise
    mesh.edges = mesh.edges[:, ::-1]

    # Solve
    streamFunctionSolver = sf.StreamFunctionSolver(mesh, airfoil_edge_id, farfield_edge_id)
    psi, c3, info = streamFunctionSolver.solve(alpha=alpha_rad)

    Cl, _, _ = streamFunctionSolver.lift_coefficient_from_circulation(psi, 1.0, 1.0)
    _, Clp, _ = streamFunctionSolver.force_coefficients(psi, 1.0)

    # write output (still needed for visualization / postprocessing, not for
    # adaptation itself now that the metric is computed directly below)
    solution_file_data = {"version": 3, "dim": 2,
                           "sol_at_vertices": {"values": np.column_stack([psi])}}
    lm.write(f"{project}_{adapt_step}.solb", solution_file_data)

    run_time = time.perf_counter() - start_time
    # streamFunctionSolver, psi, and info are returned so the adjoint-based metric
    # can be built from this exact solve without re-solving the primal problem
    return (mesh.n_nodes, Cl, Clp, c3, run_time, streamFunctionSolver, psi, info)


def compute_metric(project, adapt_step, solver, psi, info, complexity):
    """Compute the adjoint-based (goal-oriented) metric tensor for adapting the
    mesh to Cl, and write it in the symmetric-matrix GMF convention expected by
    `ref adapt --metric` (2D: 3 components per node, ordered m11, m12, m22).

    This replaces the `ref multiscale` feature-based metric computation with
    the lift-adjoint metric from `lift_adjoint_metric.py`.
    """
    start_time = time.perf_counter()
    metric_file = f"{project}_{adapt_step}_metric.solb"

    M_go, lam_full = lift_adjoint_metric(
        solver, psi, info,
        u_inf=1.0, chord=1.0,
        q=1.0, target_complexity=complexity,
        lumped=True
    )

    # (n_nodes, 2, 2) -> (n_nodes, 3) in GMF symmetric-matrix order (m11, m12, m22)
    metric_voigt = np.column_stack([M_go[:, 0, 0], M_go[:, 0, 1], M_go[:, 1, 1]])

    metric_file_data = {"version": 3, "dim": 2,
                         "sol_at_vertices": {"values": metric_voigt}}
    lm.write(metric_file, metric_file_data)

    run_time = time.perf_counter() - start_time
    return run_time


def adapt(project, adapt_step):
    start_time = time.perf_counter()
    mesh_file = f"{project}_{adapt_step}.meshb"
    metric_file = f"{project}_{adapt_step}_metric.solb"
    egads_file = f"{project}_.egads"

    adapted_mesh_file = f"{project}_{adapt_step + 1}.meshb"

    # Adapt directly from the adjoint-based metric computed in compute_metric()
    # (no `ref multiscale` call -- that metric is no longer used)
    adapt_str = f"ref adapt {mesh_file} --egads {egads_file} --metric {metric_file} -x {adapted_mesh_file} > adapt_{adapt_step}.out"

    subprocess.run(
        adapt_str,
        shell=True,
        check=True,
    )

    run_time = time.perf_counter() - start_time
    return run_time


#%% main
project = "naca0012"

start_cycle = 21
n_total_cycles = 25

start_complexity = 100
complexity_modulus = 3  # cycles before doubling

columns = ["cycle", "nNodes", "complexity", "cl", "clp", "c3", "solve_time", "metric_time", "adapt_time"]
path = "adapt_history.csv"

if start_cycle == 0:
    # Copy CAD and initial mesh from geometry folder
    shutil.copy(os.path.join("..", "geometry", f"{project}_.egads"), f"{project}_.egads")
    shutil.copy(os.path.join("..", "geometry", f"{project}_-vol.meshb"), f"{project}_0.meshb")

    # Setup log table
    df = pd.DataFrame(columns=columns)

else:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")
    df = pd.read_csv(path)

# Figure out where to resume if start_cycle < 0
if start_cycle < 0:
    start_cycle = int(df["cycle"].max()) + 1

# Set complexity
if start_cycle == 0:
    complexity = start_complexity
else:
    complexity = int(df["complexity"].max())

# Solve-Adapt cycle
print("     cycle,    nNodes,        cl,       clp,        c3,      solve_time,     metric_time,      adapt_time")
for cycle in range(start_cycle, n_total_cycles + 1):

    # Increment complexity
    if np.mod(cycle, complexity_modulus) == 0:
        complexity = complexity * 2

    # Solve, compute the adjoint-based metric, and adapt
    n_nodes, cl, clp, c3, solve_time, solver, psi, info = solve(project, cycle)
    metric_time = compute_metric(project, cycle, solver, psi, info, complexity)
    adapt_time = adapt(project, cycle)

    # Logging
    new_row = {
        "cycle": cycle,
        "nNodes": n_nodes,
        "complexity": complexity,
        "cl": cl,
        "clp": clp,
        "c3": c3,
        "solve_time": solve_time,
        "metric_time": metric_time,
        "adapt_time": adapt_time,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)  # save after every cycle

    line = f"{cycle:10d},{n_nodes:10d},{cl:10.3f},{clp:10.3f},{c3:10.3f},{solve_time:16.3f},{metric_time:16.3f},{adapt_time:16.3f}"
    print(line)
