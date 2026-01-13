# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Warp GPU Kernels for Model Order Reduction

Provides GPU-accelerated operations for reduced-order simulation:
1. Projection kernels: Full ↔ Reduced space transformations
2. SolverReduced: Reduced-order implicit solver
3. Force evaluation with hyperreduction (ECSW)

Usage:
    from warp.reduction import SolverReduced, load_reduced_model
    
    # Load pre-computed reduced basis
    reduced_data = load_reduced_model('reduced_basis/')
    
    # Create reduced solver
    solver = SolverReduced(model, reduced_data)
    
    # Simulation loop (same interface as full solver)
    for step in range(n_steps):
        solver.step(state_in, state_out, dt)
        state_in, state_out = state_out, state_in
"""

from .kernels_projection import (
    project_to_reduced_space,
    project_to_full_space,
    project_forces_to_reduced,
)

from .solver_reduced import SolverReduced

from .utils import load_reduced_model, create_reduced_arrays

__all__ = [
    'project_to_reduced_space',
    'project_to_full_space', 
    'project_forces_to_reduced',
    'SolverReduced',
    'load_reduced_model',
    'create_reduced_arrays',
]

