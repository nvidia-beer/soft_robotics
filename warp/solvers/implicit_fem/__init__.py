# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Fully Implicit FEM solver for 2D spring-mass systems
# Both springs AND FEM elements are treated implicitly

from .solver_implicit_fem import SolverImplicitFEM
from .kernels_fem_2d_implicit import (
    compute_fem_stiffness_blocks_2d,
    build_combined_system_matrix_2d,
    eval_triangles_fem_implicit_2d,
    eval_triangle_fem_forces_implicit_2d,
)

__all__ = [
    "SolverImplicitFEM",
    "compute_fem_stiffness_blocks_2d",
    "build_combined_system_matrix_2d",
    "eval_triangles_fem_implicit_2d",
    "eval_triangle_fem_forces_implicit_2d",
]

















