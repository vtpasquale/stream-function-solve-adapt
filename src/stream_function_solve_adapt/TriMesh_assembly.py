#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriMesh_assembly.py

Fast, vectorized assembly of global FEM matrices for linear (CST) triangle
elements on a TriMesh:

  - M    : consistent mass matrix          (L2 inner product of shape functions)
  - M_L  : lumped mass matrix (diagonal)
  - K    : stiffness matrix                (Laplacian / Poisson operator)
  - Cx, Cy : gradient-projection coupling matrices
  - Px, Py : nodal gradient-recovery projection operators (d/dx, d/dy)

Everything is assembled with NumPy array broadcasting (no per-element Python
loop) plus scipy.sparse.coo_matrix, which sums duplicate (row, col) entries
automatically -- that summation *is* the FEM assembly step, so no manual
scatter-add loop is required for the matrices.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from stream_function_solve_adapt.TriMesh import TriMesh 

# ---------------------------------------------------------------------------
#  Element geometry (vectorized over ALL triangles simultaneously)
# ---------------------------------------------------------------------------

def triangle_geometry(mesh: TriMesh):
    """Constant-strain-triangle (CST) geometry for every element, computed
    for the whole mesh in one shot (no Python loop over elements).

    Linear shape functions on a triangle are
        N_i(x, y) = (a_i + b_i*x + c_i*y) / (2*A)
    with the standard CST coefficients b_i, c_i built from node coordinates,
    and A the (signed) triangle area.

    Returns
    -------
    area : (M,) ndarray
        Element areas (positive for counter-clockwise node ordering).
    dNdx, dNdy : (M, 3) ndarray
        Constant shape-function derivatives for local nodes 0, 1, 2.
    """
    tri = mesh.triangles
    xy = mesh.nodes[:, :2]                        # planar mesh: use (x, y)

    x = xy[tri, 0]                                 # (M, 3)
    y = xy[tri, 1]                                 # (M, 3)

    # b_i = y_{i+1} - y_{i+2},  c_i = x_{i+2} - x_{i+1}   (cyclic local index)
    b = np.column_stack([y[:, 1] - y[:, 2],
                          y[:, 2] - y[:, 0],
                          y[:, 0] - y[:, 1]])
    c = np.column_stack([x[:, 2] - x[:, 1],
                          x[:, 0] - x[:, 2],
                          x[:, 1] - x[:, 0]])

    # Shoelace formula: 2A = x0(y1-y2) + x1(y2-y0) + x2(y0-y1)
    twice_area = (x[:, 0] * (y[:, 1] - y[:, 2]) +
                  x[:, 1] * (y[:, 2] - y[:, 0]) +
                  x[:, 2] * (y[:, 0] - y[:, 1]))

    area = 0.5 * twice_area
    dNdx = b / twice_area[:, None]
    dNdy = c / twice_area[:, None]
    return area, dNdx, dNdy


# ---------------------------------------------------------------------------
#   Generic scatter-assembly helper
# ---------------------------------------------------------------------------

def _assemble(local_blocks, tri, n_nodes):
    """Scatter a stack of dense per-element 3x3 matrices (or 3-vectors) into
    a global sparse matrix (or dense vector).

    local_blocks : (M, 3, 3) or (M, 3) ndarray
        Per-element local matrix/vector, local node ordering (0, 1, 2).
    tri : (M, 3) int ndarray
        Global node index for each local node of each element.
    n_nodes : int
        Total number of mesh nodes (global matrix/vector size).
    """
    if local_blocks.ndim == 3:
        rows = np.repeat(tri, 3, axis=1)           # i i i j j j k k k
        cols = np.tile(tri, (1, 3))                # i j k i j k i j k
        data = local_blocks.reshape(-1)
        mat = sparse.coo_matrix((data, (rows.reshape(-1), cols.reshape(-1))),
                                 shape=(n_nodes, n_nodes))
        return mat.tocsr()
    else:  # vector scatter (e.g. lumped-mass diagonal contributions)
        vec = np.zeros(n_nodes)
        np.add.at(vec, tri.reshape(-1), local_blocks.reshape(-1))
        return vec


# ---------------------------------------------------------------------------
#   Global matrices
# ---------------------------------------------------------------------------

def stiffness_matrix(mesh: TriMesh, area=None, dNdx=None, dNdy=None):
    """Global Laplacian stiffness matrix  K = sum_e A_e * B_e^T B_e."""
    if area is None:
        area, dNdx, dNdy = triangle_geometry(mesh)
    K_local = area[:, None, None] * (
        dNdx[:, :, None] * dNdx[:, None, :] +
        dNdy[:, :, None] * dNdy[:, None, :]
    )
    return _assemble(K_local, mesh.triangles, mesh.n_nodes)


def consistent_mass_matrix(mesh: TriMesh, area=None):
    """Global consistent mass matrix M = sum_e (A_e/12) * [[2,1,1],[1,2,1],[1,1,2]]."""
    if area is None:
        area, _, _ = triangle_geometry(mesh)
    template = np.array([[2., 1., 1.],
                          [1., 2., 1.],
                          [1., 1., 2.]])
    M_local = (area[:, None, None] / 12.0) * template
    return _assemble(M_local, mesh.triangles, mesh.n_nodes)


def lumped_mass_matrix(mesh: TriMesh, area=None):
    """Diagonal (row-sum lumped) mass matrix, returned as a sparse diagonal
    matrix so it composes with M / K without special-casing downstream code.
    Row-sum lumping distributes A_e/3 to each vertex of a linear triangle.
    """
    if area is None:
        area, _, _ = triangle_geometry(mesh)
    diag_contrib = np.repeat(area[:, None] / 3.0, 3, axis=1)    # (M, 3)
    diag = _assemble(diag_contrib, mesh.triangles, mesh.n_nodes)
    return sparse.diags(diag)


def gradient_coupling_matrices(mesh: TriMesh, area=None, dNdx=None, dNdy=None):
    """Global coupling matrices Cx, Cy such that

        M @ (dphi/dx)~ = Cx @ phi~ ,   M @ (dphi/dy)~ = Cy @ phi~

    Elementwise: C_i[a, b] = (A_e / 3) * dN_b/dx_i, independent of the row
    index a, because linear shape functions integrate to A_e/3 over the
    element and the derivative dN_b/dx_i is constant per element.
    """
    if area is None:
        area, dNdx, dNdy = triangle_geometry(mesh)
    ones_col = np.ones((area.shape[0], 3, 1))
    Cx_local = (area[:, None, None] / 3.0) * ones_col * dNdx[:, None, :]
    Cy_local = (area[:, None, None] / 3.0) * ones_col * dNdy[:, None, :]
    Cx = _assemble(Cx_local, mesh.triangles, mesh.n_nodes)
    Cy = _assemble(Cy_local, mesh.triangles, mesh.n_nodes)
    return Cx, Cy


# -----------------------------------------------------------------------------
#   Element gradient recovery - directly from nodal values and shape functions
# -----------------------------------------------------------------------------

def element_gradient_operators(mesh: TriMesh, dNdx=None, dNdy=None):
    """Build sparse (n_triangles, n_nodes) operators Gx, Gy such that

        dphi/dx|_e = Gx @ phi ,   dphi/dy|_e = Gy @ phi

    Row e has three nonzeros: dNdx[e, a] placed at column mesh.triangles[e, a].
    This is the discrete gradient (B) operator from isoparametric theory --
    it maps nodal values directly to the piecewise-constant element gradient,
    with no averaging or mass-matrix solve involved.
    """
    if dNdx is None:
        _, dNdx, dNdy = triangle_geometry(mesh)
    tri = mesh.triangles
    n_tri = tri.shape[0]
    rows = np.repeat(np.arange(n_tri), 3)
    cols = tri.reshape(-1)
    Gx = sparse.csr_matrix((dNdx.reshape(-1), (rows, cols)), shape=(n_tri, mesh.n_nodes))
    Gy = sparse.csr_matrix((dNdy.reshape(-1), (rows, cols)), shape=(n_tri, mesh.n_nodes))
    return Gx, Gy


def compute_element_gradients(mesh: TriMesh, phi, Gx=None, Gy=None):
    """Evaluate element-wise gradients for a nodal field phi.

    Returns an (n_triangles, 2) array with columns [dphi/dx, dphi/dy].
    Pass precomputed Gx, Gy to avoid rebuilding them when phi changes but
    the mesh does not (e.g. inside a solve or adaptation loop).
    """
    if Gx is None:
        Gx, Gy = element_gradient_operators(mesh)
    return np.column_stack([Gx @ phi, Gy @ phi])

# ---------------------------------------------------------------------------
#   Nodal gradient recovery L2 projection
# ---------------------------------------------------------------------------

def gradient_projection_operators(mesh: TriMesh, lumped=False):
    """Build sparse operators Px, Py such that, for nodal values phi~,

        (dphi/dx)~ = Px @ phi~ ,   (dphi/dy)~ = Py @ phi~

    lumped=True  -> M_L^-1 @ C   (fast diagonal solve, ~1 order less accurate)
    lumped=False -> M^-1  @ C    (consistent L2 projection; one sparse LU
                                   factorization of M is reused for both Px
                                   and Py)
    """
    area, dNdx, dNdy = triangle_geometry(mesh)
    Cx, Cy = gradient_coupling_matrices(mesh, area, dNdx, dNdy)

    if lumped:
        M_L_inv = sparse.diags(1.0 / lumped_mass_matrix(mesh, area).diagonal())
        Px, Py = M_L_inv @ Cx, M_L_inv @ Cy
    else:
        solve = sparse.linalg.factorized(consistent_mass_matrix(mesh, area).tocsc())
        Px = sparse.csr_matrix(solve(Cx.toarray()))
        Py = sparse.csr_matrix(solve(Cy.toarray()))
    return Px, Py


def compute_gradients(mesh: TriMesh, phi, lumped=False):
    """Evaluate nodal gradient components for a scalar nodal field phi.

    (dphi/dx)~ = Px @ phi , (dphi/dy)~ = Py @ phi

    Returns an (n_nodes, 2) array with columns [dphi/dx, dphi/dy].
    """
    Px, Py = gradient_projection_operators(mesh, lumped=lumped)
    gx, gy = Px @ phi, Py @ phi
    return np.column_stack([gx, gy])


# ---------------------------------------------------------------------------
#   Hessian recovery (L2 projection)
# ---------------------------------------------------------------------------

def hessian_projection_operators(mesh: TriMesh, lumped=False):
    """Build the three symmetrized second-derivative operators.

    For nodal values phi, the recovered Hessian components are
        Hxx~ = Hxx_op @ phi,  Hxy~ = Hxy_op @ phi,  Hyy~ = Hyy_op @ phi

    Cross term is symmetrized: Hxy_op = 0.5*(Px@Py + Py@Px), since the two
    application orders don't commute exactly at the discrete level.
    """
    Px, Py = gradient_projection_operators(mesh, lumped=lumped)
    Hxx_op = Px @ Px
    Hyy_op = Py @ Py
    Hxy_op = 0.5 * (Px @ Py + Py @ Px)
    return Hxx_op, Hxy_op, Hyy_op


def compute_hessian(mesh: TriMesh, phi, lumped=False):
    """Evaluate nodal Hessian components for a scalar nodal field phi.

    Returns an (n_nodes, 3) array in Voigt order [Hxx, Hyy, Hxy].
    """
    Hxx_op, Hxy_op, Hyy_op = hessian_projection_operators(mesh, lumped=lumped)
    Hxx, Hxy, Hyy = Hxx_op @ phi, Hxy_op @ phi, Hyy_op @ phi
    return np.column_stack([Hxx, Hyy, Hxy])

def compute_hessian_fast(mesh: TriMesh, phi, lumped=False):
    
    area, dNdx, dNdy = triangle_geometry(mesh)
    Cx, Cy = gradient_coupling_matrices(mesh, area, dNdx, dNdy)
    
    if lumped:
        M = lumped_mass_matrix(mesh, area)
    else:
        M = consistent_mass_matrix(mesh, area)

    lu = splu(M.tocsc()) # one sparse LU factorization, reused for every solve
    gx = lu.solve(Cx @ phi)        # grad_x, via a sparse matvec + one triangular solve
    gy = lu.solve(Cy @ phi)        # grad_y
    Hxx = lu.solve(Cx @ gx)         # second derivative, same pattern applied to gx
    Hyy = lu.solve(Cy @ gy)
    Hxy = 0.5 * (lu.solve(Cx @ gy) + lu.solve(Cy @ gx))
    return np.column_stack([Hxx, Hyy, Hxy])


def voigt_to_tensor(H_voigt):
    """(N,3) [Hxx,Hyy,Hxy] -> (N,2,2) symmetric tensor, built only when needed
    (e.g., immediately before eigendecomposition for a metric field)."""
    Hxx, Hyy, Hxy = H_voigt[:, 0], H_voigt[:, 1], H_voigt[:, 2]
    T = np.zeros((H_voigt.shape[0], 2, 2))
    T[:, 0, 0], T[:, 1, 1] = Hxx, Hyy
    T[:, 0, 1] = T[:, 1, 0] = Hxy
    return T

# # ---------------------------------------------------------------------------
# # 5. Convenience: build everything in a single pass
# # ---------------------------------------------------------------------------

# @dataclass
# class FemOperators:
#     K: sparse.csr_matrix
#     M: sparse.csr_matrix
#     M_L: sparse.dia_matrix
#     Cx: sparse.csr_matrix
#     Cy: sparse.csr_matrix
#     Px: sparse.csr_matrix
#     Py: sparse.csr_matrix


# def build_operators(mesh: TriMesh, lumped_projection=True):
#     """Assemble the full operator set, computing element geometry only once."""
#     area, dNdx, dNdy = triangle_geometry(mesh)

#     K = stiffness_matrix(mesh, area, dNdx, dNdy)
#     M = consistent_mass_matrix(mesh, area)
#     M_L = lumped_mass_matrix(mesh, area)
#     Cx, Cy = gradient_coupling_matrices(mesh, area, dNdx, dNdy)

#     if lumped_projection:
#         M_L_inv = sparse.diags(1.0 / M_L.diagonal())
#         Px, Py = M_L_inv @ Cx, M_L_inv @ Cy
#     else:
#         solve = sparse.linalg.factorized(M.tocsc())
#         Px = sparse.csr_matrix(solve(Cx.toarray()))
#         Py = sparse.csr_matrix(solve(Cy.toarray()))

#     return FemOperators(K=K, M=M, M_L=M_L, Cx=Cx, Cy=Cy, Px=Px, Py=Py)


# if __name__ == "__main__":
#     # Minimal smoke test: a two-triangle unit square, no external mesh file.

#     nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
#     triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
#     mesh = TriMesh(nodes, triangles)

#     ops = build_operators(mesh)
#     print("K row sums (should be ~0):", np.array(ops.K.sum(axis=1)).ravel())
#     print("Total mass (should equal area = 1):", ops.M.sum())

#     # Gradient recovery sanity check: phi = 2x + 3y -> grad = (2, 3) everywhere
#     phi = 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
#     Px_c, Py_c = gradient_projection_operators(mesh, lumped=False)
#     Px_l, Py_l = gradient_projection_operators(mesh, lumped=True)
#     print("consistent grad:", Px_c @ phi, Py_c @ phi)
#     print("lumped grad:    ", Px_l @ phi, Py_l @ phi)
