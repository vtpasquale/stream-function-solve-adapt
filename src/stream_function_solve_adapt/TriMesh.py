#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:58:41 2026

@author: vtpasquale
"""

from dataclasses import dataclass, field
import numpy as np
import pyLibMeshb.libMeshb as lm

@dataclass
class TriMesh:
    """
    Triangular mesh with implicit sequential numbering.

    Node and element identity is purely positional (array index) — no
    Nastran/SU2 IDs are stored. CBEAM/SU2-boundary elements are stored as
    edges (node-index pairs) with a property/marker ID retained as an
    edge/boundary ID.

    Attributes
    ----------
    nodes : np.ndarray, shape (N, 3)
        Node coordinates. Node i is referenced implicitly by index i.
    triangles : np.ndarray, shape (M, 3), dtype int
        Triangle connectivity as node indices into `nodes` (0-based).
    edges : np.ndarray, shape (E, 2), dtype int
        Boundary/beam connectivity as node indices into `nodes` (0-based).
    edge_ids : np.ndarray, shape (E,), dtype int
        Boundary/property ID per edge, positionally aligned with `edges`.
    """
    nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    edge_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    # ---------- Constructor  ----------

    @classmethod
    def from_file(cls, path: str) -> "TriMesh":
        
        # Read mesh data from file
        mesh = lm.read(path)
        
        # Confirm required keys are available
        required_keys = ['vertices', 'edges', 'triangles']
        missing = [k for k in required_keys if k not in mesh]
        if missing:
            raise KeyError(f"Missing required key(s) in mesh data: {', '.join(missing)}")

        # Data conversion
        return TriMesh(mesh["vertices"],mesh["triangles"][:,0:3]-1,mesh["edges"][:,0:2]-1,mesh["edges"][:,2])

    # ---------- Convenience properties ----------

    @property
    def n_nodes(self) -> int: return self.nodes.shape[0]
    @property
    def n_triangles(self) -> int: return self.triangles.shape[0]
    @property
    def n_edges(self) -> int: return self.edges.shape[0]

    # # ---------- Mesh editing helpers ----------

    def summary(self) -> str:
        return f"TriMesh: {self.n_nodes} nodes, {self.n_triangles} triangles, {self.n_edges} edges"

    def __repr__(self) -> str:
        return self.summary()    
    