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
import pyLibMeshb.libMeshb as lm


def solve(project,adapt_step):
    start_time = time.perf_counter() 
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
    
    run_time = time.perf_counter() - start_time
    return (mesh.n_nodes, Cl, c3, run_time)

def adapt(project, adapt_step, complexity):
    start_time = time.perf_counter() 
    mesh_file = f"{project}_{adapt_step}.meshb"
    solution_file = f"{project}_{adapt_step}.solb"
    metric_file = f"{project}_{adapt_step}_metric.solb"
    egads_file = f"{project}_.egads"
    
    adapted_mesh_file = f"{project}_{adapt_step+1}.meshb"
    
    # Compute metric
    multiscale_str = f"ref multiscale {mesh_file} {solution_file} {complexity} {metric_file} > multiscale_{adapt_step}.out"
    
    subprocess.run(
        multiscale_str,
        shell=True,
        check=True,
    )

    # Adapt
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

start_cycle = -1
n_total_cycles = 25

start_complexity = 100
complexity_modulus = 3 # cycles before doubling 

columns = ["cycle", "nNodes", "complexity", "cl", "c3", "solve_time", "adapt_time"]
path = "adapt_history.csv"

if start_cycle == 0:
    # Copy CAD and initial mesh from geometry folder
    shutil.copy(os.path.join("..","geometry",f"{project}_.egads"), f"{project}_.egads")
    shutil.copy(os.path.join("..","geometry",f"{project}_-vol.meshb"), f"{project}_0.meshb")
    
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
print("     cycle,    nNodes,        cl,        c3,      solve_time,      adapt_time")
for cycle in range(start_cycle, n_total_cycles + 1):
    
    # Increment complexity
    if np.mod(cycle,complexity_modulus) == 0:
        complexity = complexity*2

    # Solve and adapt
    n_nodes, cl, c3, solve_time = solve(project,cycle)
    adapt_time = adapt(project,cycle,complexity)
    
    # Logging
    new_row = {
        "cycle": cycle,
        "nNodes": n_nodes,
        "complexity": complexity,
        "cl": cl,
        "c3": c3,
        "solve_time": solve_time,
        "adapt_time": adapt_time,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)   # save after every cycle
    
    line = f"{cycle:10d},{n_nodes:10d},{cl:10.3f},{c3:10.3f},{solve_time:16.3f},{adapt_time:16.3f}"
    print(line)
    
    

# n_steps = 30
# complexity = 100

# header = "     cycle,    nNodes,        cl,        c3,      solve_time,      adapt_time"
# with open(f"{project}.log","w") as table_fid:
#     print(header)
#     print(header,file=table_fid)

#     for i in range(0,n_steps): 
        
#         if np.mod(i,6) == 0:
#             complexity = complexity*2
        

        
#         # log result
#         line = f"{i:10d},{n_nodes:10d},{cl:10.3f},{c3:10.3f},{solve_time:16.3f},{adapt_time:16.3f}"
#         print(line)
#         print(line,file=table_fid)



# metric = lm.read("metric.solb")
# lm.mesh_info("metric.solb")