# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Vertex Block Descent (VBD) solver for 2D spring-mass / FEM simulations
# Based on: Chen et al. 2024, "Vertex Block Descent" (SIGGRAPH)
# Adapted from WarpVBD 3D implementation

from .solver_vbd import SolverVBD
from .coloring import graph_coloring_2d, compute_adjacency_from_triangles

__all__ = [
    "SolverVBD",
    "graph_coloring_2d",
    "compute_adjacency_from_triangles",
]
