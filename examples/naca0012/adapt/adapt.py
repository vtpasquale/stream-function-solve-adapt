# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:31:31 2026

@author: aricciar
"""
import subprocess
import numpy as np
import stream_function_solve_adapt.TriMesh as tm
import stream_function_solve_adapt.solve as sf

import pyLibMeshb.libMeshb as lm


def solve(project,adapt_step):
    mesh_file = f"{project}_{adapt_step}.meshb"
    alpha_rad = float( np.deg2rad(10.0) )
    
    # Read mesh and update edge ids
    mesh = tm.TriMesh.from_file(mesh_file)
    
    # Update edge ids
    airfoil_edge_id = 0
    farfield_edge_id= 1
    mesh.edge_ids[mesh.edge_ids==2] = farfield_edge_id
    mesh.edge_ids[mesh.edge_ids==3] = airfoil_edge_id
    mesh.edge_ids[mesh.edge_ids==4] = airfoil_edge_id
    
    streamFunctionSolver = sf.StreamFunctionSolver(mesh, airfoil_edge_id, farfield_edge_id)
    psi, c3, info = streamFunctionSolver.solve(alpha = alpha_rad)
    
    Cl, _, _ = streamFunctionSolver.lift_coefficient_from_circulation(psi, 1.0, 1.0)
    
    
    # write output
    solution_file_data = {"version": 3, "dim": 2, 
                          "sol_at_vertices": {"values": np.column_stack([psi])} }
    lm.write(f"{project}_{adapt_step}.solb",solution_file_data)
    
    return (Cl, c3, mesh.n_nodes)

def adapt(project, adapt_step, complexity):
    mesh_file = f"{project}_{adapt_step}.meshb"
    solution_file = f"{project}_{adapt_step}.solb"
    metric_file = f"{project}_{adapt_step}_metric.solb"
    egads_file = f"{project}_.egads"
    
    adapted_mesh_file = f"{project}_{adapt_step+1}.meshb"
    
    # Compute metric
    multiscale_str = f"ref multiscale {mesh_file} {solution_file} {complexity} {metric_file} > multiscale_{adapt_step}.out"
    
    subprocess.run(
        multiscale_str,       # command as a list of strings, not a single string
        capture_output=True, # capture stdout/stderr
        shell=True,
        check=True,          # raise CalledProcessError if the command fails
    )

    # Adapt
    adapt_str = f"ref adapt {mesh_file} --egads {egads_file} --metric {metric_file} -x {adapted_mesh_file} > adapt_{adapt_step}.out"
    
    subprocess.run(
        adapt_str,       # command as a list of strings, not a single string
        capture_output=True, # capture stdout/stderr
        shell=True,
        check=True,          # raise CalledProcessError if the command fails
    )
    

#%%
project = "naca0012"

n_steps = 25
complexity = 500


cl = np.zeros(n_steps)
c3 = np.zeros(n_steps)
n_nodes = np.zeros(n_steps)

for i in range(0,n_steps): 
    
    if np.mod(i,5) == 0:
        complexity = complexity*2

    cl[i], c3[i], n_nodes[i] = solve(project,i)
    adapt(project,i,complexity)



# metric = lm.read("metric.solb")
# lm.mesh_info("metric.solb")