# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Implicit solver for 2D spring-mass systems with FEM support

from .solver_implicit import SolverImplicit
from .kernels_fem_2d import (
    eval_triangles_fem_2d,
    eval_springs_2d,
    build_system_matrix_sparse_2d,
    update_state_2d,
    eval_gravity_2d,
    apply_boundary_2d,
    apply_boundary_with_friction_2d,
    eval_spring_forces_2d,
    eval_triangle_fem_forces_2d,
)

__all__ = [
    "SolverImplicit",
    "eval_triangles_fem_2d",
    "eval_springs_2d",
    "build_system_matrix_sparse_2d",
    "update_state_2d",
    "eval_gravity_2d",
    "apply_boundary_2d",
    "apply_boundary_with_friction_2d",
    "eval_spring_forces_2d",
    "eval_triangle_fem_forces_2d",
]
