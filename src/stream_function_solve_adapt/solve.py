"""
stream_function_solver.py

Implements the stream-function finite element formulation described in
streamFunctionAnalysis_v20241110.docx (A. Ricciardi, May 2024) on top of
a TriMesh (nodes / triangles / edges / edge_ids).

Governing equation:   Laplace(psi) = 0                          (Eq. 10)
Boundary conditions:  psi_a = -u2inf*x1 + u1inf*x2 + c3          (Eq. 11)
                       psi_b = 0                                  (Eq. 12)
Constraint equation:  grad(psi_te) . n_te = 0                     (Eq. 16)

The unknown constant c3 is retained as an extra system unknown and solved
for simultaneously with the nodal psi values via a bordered linear system
(Eq. 36), following the document's derivation.

Kutta-constraint direction (IMPORTANT):
    The document's stream-function convention is grad(psi) = R u, where R
    is a 90-degree rotation (Eq. 1-3). This means grad(psi) is ALWAYS
    perpendicular to the velocity it represents. Consequently, requiring
    velocity to be parallel to the trailing-edge bisector direction t
    translates to:

        grad(psi_te) . t  =  0        (dot with the BISECTOR itself)

    NOT grad(psi_te) . n = 0 where n is the bisector rotated 90 degrees.
    (If u is parallel to t, then grad(psi) = R u is parallel to R t = n,
    i.e. grad(psi) has ZERO component along t and a NONZERO component
    along n -- so the correct zero-valued dot product uses t, not n.)
    This module implements the corrected form throughout.

Wake-element selection (IMPORTANT):
    The trailing-edge element used to evaluate grad(psi_te) must have at
    MOST ONE node on the airfoil boundary. Elements with two or more
    airfoil-boundary nodes are surface-following triangles, not wake
    elements; using one degenerates the constraint row to a single
    coefficient that has nothing to do with the Kutta condition.
"""

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_array, csr_array, bmat
from scipy.sparse.linalg import spsolve


class StreamFunctionSolver:
    """Stream-function FE solver for 2D lifting-body potential flow.

    Parameters
    ----------
    mesh : TriMesh
        Mesh with `nodes` (N,3), `triangles` (M,3), `edges` (E,2),
        `edge_ids` (E,).
    airfoil_edge_id : int
        `edge_ids` value marking the airfoil (body) boundary edges.
    farfield_edge_id : int
        `edge_ids` value marking the farfield boundary edges.
    """

    def __init__(self, mesh, airfoil_edge_id, farfield_edge_id):
        self.mesh = mesh
        self.airfoil_edge_id = airfoil_edge_id
        self.farfield_edge_id = farfield_edge_id
        self.coords = mesh.nodes[:, :2]
        self.conn = mesh.triangles
        self.n_nodes = mesh.n_nodes
        self._geometry()
        self._boundary_sets()

    # ------------------------------------------------------------------
    # Geometry and element-level quantities (Eqs. 17-24)
    # ------------------------------------------------------------------
    def _geometry(self):
        x = self.coords[self.conn]
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        area2 = ((x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1])
                 - (x3[:, 0] - x1[:, 0]) * (x2[:, 1] - x1[:, 1]))
        self.area = np.abs(area2) / 2.0
        self.b = np.stack([x2[:, 1] - x3[:, 1],
                            x3[:, 1] - x1[:, 1],
                            x1[:, 1] - x2[:, 1]], axis=1)
        self.c = np.stack([x3[:, 0] - x2[:, 0],
                            x1[:, 0] - x3[:, 0],
                            x2[:, 0] - x1[:, 0]], axis=1)
        if np.any(self.area <= 0):
            raise ValueError("Non-positive element area; check connectivity winding.")

    def _rows_cols_33(self):
        rows = np.repeat(self.conn, 3, axis=1)
        cols = np.tile(self.conn, (1, 3))
        return rows, cols

    def stiffness(self):
        """Global stiffness matrix, Eq. (24): K = sum_e (1/4A)(b b^T + c c^T)."""
        Ke = (np.einsum('ei,ej->eij', self.b, self.b)
              + np.einsum('ei,ej->eij', self.c, self.c)) / (4 * self.area)[:, None, None]
        rows, cols = self._rows_cols_33()
        return coo_array((Ke.reshape(-1, 9).ravel(), (rows.ravel(), cols.ravel())),
                          shape=(self.n_nodes, self.n_nodes)).tocsr()

    def gradient_operator(self):
        """Elementwise-constant gradient operator, shape (2*n_elem, n_nodes)."""
        Be = np.stack([self.b, self.c], axis=1) / (2 * self.area)[:, None, None]
        n_elem = self.conn.shape[0]
        eidx = np.arange(n_elem)
        rows = np.repeat(2 * eidx[:, None], 3, axis=1)
        rows = np.stack([rows, rows + 1], axis=1).reshape(n_elem, 2, 3)
        cols = np.repeat(self.conn[:, None, :], 2, axis=1)
        return coo_array((Be.ravel(), (rows.ravel(), cols.ravel())),
                          shape=(2 * n_elem, self.n_nodes)).tocsr()

    # ------------------------------------------------------------------
    # Boundary identification
    # ------------------------------------------------------------------
    def _boundary_sets(self):
        af = self.mesh.edge_ids == self.airfoil_edge_id
        ff = self.mesh.edge_ids == self.farfield_edge_id
        self.airfoil_edges = self.mesh.edges[af]
        self.farfield_edges = self.mesh.edges[ff]
        self.airfoil_nodes = np.unique(self.airfoil_edges)
        self.farfield_nodes = np.unique(self.farfield_edges)

    def _airfoil_loop(self):
        """Order the airfoil boundary nodes into a single closed loop."""
        adj = defaultdict(list)
        for a, b in self.airfoil_edges:
            adj[a].append(b)
            adj[b].append(a)
        start = self.airfoil_edges[0, 0]
        loop, prev, cur = [start], None, start
        while True:
            nbrs = [n for n in adj[cur] if n != prev]
            nxt = nbrs[0] if nbrs else None
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        return loop

    def detect_te_node(self, chord_dir=(1.0, 0.0)):
        """Automatically identify the trailing-edge node.

        Uses the airfoil-loop node with the largest projection onto the
        known chordwise direction. This is robust for sharp, shallow, or
        near-cusped trailing edges, unlike curvature/interior-angle based
        corner detection, which can mis-fire on cusped TEs (where the
        included angle is close to 180 degrees) or on coarsely resolved
        high-curvature noses.
        """
        loop = self._airfoil_loop()
        chord_dir = np.asarray(chord_dir, dtype=float)
        proj = self.coords[loop] @ chord_dir
        te_node = loop[int(np.argmax(proj))]
        return te_node, loop

    def auto_te_element_and_tangent(self, chord_dir=(1.0, 0.0)):
        """Automatically select the wake-side element and bisector tangent.

        Returns
        -------
        te_element : int
            Index into self.conn of the element used to evaluate grad(psi_te).
            Guaranteed to have at most one node on the airfoil boundary --
            elements with two or more boundary nodes are surface-following
            triangles and are explicitly excluded (see module docstring).
        t_bisector : ndarray, shape (2,)
            Unit vector along the upper/lower-surface bisection line. This
            is the vector that must be dotted with grad(psi_te) in the
            Kutta constraint (NOT its 90-degree-rotated perpendicular).
        te_node : int
            Node index of the detected trailing edge.
        """
        te_node, loop = self.detect_te_node(chord_dir)
        i = loop.index(te_node)
        n = len(loop)
        v1 = self.coords[loop[(i - 1) % n]] - self.coords[te_node]
        v2 = self.coords[loop[(i + 1) % n]] - self.coords[te_node]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        bisector = -(v1 + v2)
        norm = np.linalg.norm(bisector)
        if norm < 1e-12:
            t_bisector = np.array([v1[1], -v1[0]])   # fallback for near-collinear surfaces
        else:
            t_bisector = bisector / norm

        touching = np.where(np.any(self.conn == te_node, axis=1))[0]
        n_airfoil_in_elem = np.isin(self.conn[touching], self.airfoil_nodes).sum(axis=1)
        valid = touching[n_airfoil_in_elem <= 1]
        if len(valid) == 0:
            raise ValueError(
                f"No wake-side element found at TE node {te_node}: every element "
                f"touching it has >=2 airfoil-boundary nodes. The mesh needs at "
                f"least one element downstream of the TE with only the TE node "
                f"itself constrained for the Kutta constraint to be well-posed."
            )
        centroids = self.coords[self.conn[valid]].mean(axis=1)
        dirs = centroids - self.coords[te_node]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        te_element = valid[np.argmax(dirs @ t_bisector)]
        return te_element, t_bisector, te_node

    def _kutta_row(self, te_element, t_bisector, f_nodes):
        """Discrete Kutta constraint row: grad(psi_te) . t_bisector = 0.

        Assumes psi_s = 0 at any airfoil (constrained) node in te_element
        (true here since psi_b = 0, Eq. 12), so only free-DOF columns
        contribute to the returned row vector.
        """
        tri_nodes = self.conn[te_element]
        dN = np.stack([self.b[te_element], self.c[te_element]]) / (2 * self.area[te_element])
        coeff = t_bisector @ dN
        node_to_f = {node: i for i, node in enumerate(f_nodes)}
        g = np.zeros(len(f_nodes))
        for local_i, node in enumerate(tri_nodes):
            if node in node_to_f:
                g[node_to_f[node]] += coeff[local_i]
        return g

    # ------------------------------------------------------------------
    # Solve (Eqs. 26-36)
    # ------------------------------------------------------------------
    def solve(self, u_inf=1.0, alpha=0.0, te_element=None, t_bisector=None, chord_dir=(1.0, 0.0)):
        """Solve for psi and the unknown farfield constant c3.

        Parameters
        ----------
        u_inf, alpha : float
            Freestream speed and angle of attack (radians).
        te_element, t_bisector : optional
            If omitted, both are determined automatically via
            `auto_te_element_and_tangent`.
        chord_dir : tuple
            Chordwise reference direction used for automatic TE detection.

        Returns
        -------
        psi : ndarray, shape (n_nodes,)
        c3 : float
        info : dict with 'te_element', 't_bisector', 'te_node', 'f_nodes'
        """
        u1inf, u2inf = u_inf * np.cos(alpha), u_inf * np.sin(alpha)

        s_nodes = np.union1d(self.airfoil_nodes, self.farfield_nodes)
        f_nodes = np.setdiff1d(np.arange(self.n_nodes), s_nodes)

        psi_s_known = np.zeros(len(s_nodes))
        e_s_a = np.zeros(len(s_nodes))
        far_pos = np.searchsorted(s_nodes, self.farfield_nodes)
        x1f, x2f = self.coords[self.farfield_nodes, 0], self.coords[self.farfield_nodes, 1]
        psi_s_known[far_pos] = -u2inf * x1f + u1inf * x2f     # Eq. (11), c3 excluded
        e_s_a[far_pos] = 1.0                                    # Eq. (29)

        K = self.stiffness()
        K_ff = K[f_nodes, :][:, f_nodes].tocsr()
        K_fs = K[f_nodes, :][:, s_nodes].tocsr()

        rhs_f = -(K_fs @ psi_s_known)                           # Eq. (31), f_f = 0
        ke_hat = K_fs @ e_s_a                                    # Eq. (32)

        te_node = None
        if te_element is None or t_bisector is None:
            te_element, t_bisector, te_node = self.auto_te_element_and_tangent(chord_dir)
        g_row = self._kutta_row(te_element, t_bisector, f_nodes)

        A = bmat([[K_ff, csr_array(ke_hat.reshape(-1, 1))],
                  [csr_array(g_row.reshape(1, -1)), csr_array([[0.0]])]], format='csc')
        rhs = np.concatenate([rhs_f, [0.0]])

        sol = spsolve(A, rhs)
        psi_f, c3 = sol[:-1], sol[-1]

        psi = np.zeros(self.n_nodes)
        psi[f_nodes] = psi_f
        psi[s_nodes] = psi_s_known + c3 * e_s_a

        info = dict(te_element=te_element, t_bisector=t_bisector, te_node=te_node,
                    f_nodes=f_nodes, g_row=g_row)
        return psi, c3, info

    # ------------------------------------------------------------------
    # Output recovery (Eqs. 38-48)
    # ------------------------------------------------------------------
    def velocity_elements(self, psi):
        """Elementwise-constant velocity from Eq. (3): u1=dpsi/dx2, u2=-dpsi/dx1."""
        psi_e = psi[self.conn]
        u1 = np.einsum('ei,ei->e', self.c, psi_e) / (2 * self.area)
        u2 = -np.einsum('ei,ei->e', self.b, psi_e) / (2 * self.area)
        return np.stack([u1, u2], axis=1)

    def cp_elements(self, psi, u_inf):
        """Pressure coefficient per element, Eq. (38)."""
        u = self.velocity_elements(psi)
        V2 = (u ** 2).sum(axis=1)
        return 1.0 - V2 / u_inf ** 2

    def _edge_to_element_map(self, edges):
        edge_elem = defaultdict(list)
        for e, tri in enumerate(self.conn):
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                edge_elem[tuple(sorted((a, b)))].append(e)
        return [edge_elem.get(tuple(sorted((a, b))), [None])[0] for a, b in edges]

    def force_coefficients(self, psi, u_inf, alpha=0.0):
        """Force coefficients via surface pressure integration, Eqs. (39)-(40)."""
        cp = self.cp_elements(psi, u_inf)
        elem_of_edge = self._edge_to_element_map(self.airfoil_edges)
        Cx = np.zeros(2)
        for (n_a, n_b), e in zip(self.airfoil_edges, elem_of_edge):
            if e is None:
                continue
            p_a, p_b = self.coords[n_a], self.coords[n_b]
            tvec = p_b - p_a
            length = np.linalg.norm(tvec)
            t = tvec / length
            n_outward = np.array([t[1], -t[0]])   # rotate tangent -90 deg
            Cx += -n_outward * cp[e] * length
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        Cd, Cl = R @ Cx
        return Cd, Cl, Cx

    def circulation_row(self):
        """Row vector g_psi_gamma s.t. gamma = g_psi_gamma . psi, Eqs. (42)-(45)."""
        g = np.zeros(self.n_nodes)
        elem_of_edge = self._edge_to_element_map(self.airfoil_edges)
        for (n_a, n_b), e in zip(self.airfoil_edges, elem_of_edge):
            if e is None:
                continue
            p_a, p_b = self.coords[n_a], self.coords[n_b]
            tvec = p_b - p_a
            length = np.linalg.norm(tvec)
            t = tvec / length
            coeff = length * (t[0] * self.c[e] - t[1] * self.b[e]) / (2 * self.area[e])
            g[self.conn[e]] += coeff
        return g

    def lift_coefficient_from_circulation(self, psi, u_inf, chord):
        """Cl from circulation via Kutta-Joukowski, Eqs. (41)-(47)."""
        g = self.circulation_row()
        gamma = g @ psi
        Cl = 2 * gamma / (u_inf * chord)
        return Cl, gamma, g

    def dCl_dpsi(self, u_inf, chord):
        """Analytic sensitivity dCl/dpsi, Eq. (48). Useful as the adjoint RHS."""
        return (2.0 / (u_inf * chord)) * self.circulation_row()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def kutta_residual(self, psi, info):
        """Check how well the Kutta condition is satisfied on the solved
        field: returns (residual_velocity_component, velocity_along_wake).

        residual should be ~0 (machine precision) if the constraint was
        assembled and solved correctly. It is the velocity component
        perpendicular to the bisector -- i.e. u . n_perp, where n_perp is
        t_bisector rotated 90 degrees. This is equivalent to checking
        grad(psi_te) . t_bisector = 0 directly (the two are related by the
        R rotation, Eq. 1), but expressed in velocity terms since that is
        typically the more physically intuitive quantity to inspect.
        """
        te_element, t_bisector = info['te_element'], info['t_bisector']
        n_perp = np.array([-t_bisector[1], t_bisector[0]])
        U = self.velocity_elements(psi)
        u_te = U[te_element]
        residual = u_te @ n_perp
        along_wake = u_te @ t_bisector
        return residual, along_wake
