# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Fully Implicit solver for 2D spring-mass systems with FEM
# Both springs AND FEM elements are treated implicitly

import numpy as np
import warp as wp
from warp.optim.linear import cg, preconditioner, bicgstab, gmres, cr
from warp.sparse import BsrMatrix, bsr_zeros, bsr_set_from_triplets

from ..solver import SolverBase
from ..implicit.kernels_fem_2d import (
    eval_spring_forces_2d,
    update_state_2d,
    eval_gravity_2d,
    apply_boundary_2d,
)
from .kernels_fem_2d_implicit import (
    compute_fem_stiffness_blocks_2d,
    build_combined_system_matrix_2d,
    eval_triangle_fem_forces_implicit_2d,
)


class SolverImplicitFEM(SolverBase):
    """
    Fully Implicit solver for 2D spring-mass systems with FEM.
    
    Key Difference from SolverImplicit:
    ----------------------------------
    This solver treats BOTH springs AND FEM elements implicitly by including
    both stiffness contributions in the system matrix.
    
    System Matrix: A = M - h*D - h²*(K_spring + K_fem)
    
    Benefits:
        - Better stability for stiff FEM materials
        - Allows even larger time steps
        - More accurate for highly deformable materials
    
    Trade-offs:
        - System matrix must be rebuilt when FEM stiffness changes
        - More complex assembly (9 blocks per triangle vs 4 per spring)
        - Slightly higher per-step cost
    
    Algorithm:
    ---------
    1. Build system matrix A with both spring and FEM stiffness
    2. Compute RHS: b = h * (f_spring + f_fem + f_gravity + f_external)
    3. Solve: A * dv = b
    4. Update: v_{n+1} = v_n + dv, x_{n+1} = x_n + v_{n+1} * h
    
    The FEM tangent stiffness K_fem = ∂f_fem/∂x is computed from the
    linearization of the Neo-Hookean stress tensor around current deformation.
    
    Example:
        >>> model = Model.from_grid(N=10, spacing=0.2)
        >>> # Enable FEM on model...
        >>> solver = SolverImplicitFEM(model, dt=0.01, solver_type='bicgstab')
        >>> state_in = model.state()
        >>> state_out = model.state()
        >>> 
        >>> for i in range(1000):
        >>>     solver.step(state_in, state_out, dt=0.01)
        >>>     state_in, state_out = state_out, state_in
    """
    
    def __init__(
        self,
        model,
        dt: float = 0.05,
        mass: float = 1.0,
        preconditioner_type: str = "diag",
        solver_type: str = "bicgstab",
        max_iterations: int = 10,
        tolerance: float = 1e-3,
        rebuild_matrix_every: int = 1,
    ):
        """
        Initialize the fully implicit FEM solver.
        
        Args:
            model: The 2D Model to be simulated
            dt: Time step for integration
            mass: Particle mass (uniform)
            preconditioner_type: Type of preconditioner ("id", "diag", "diag_abs")
            solver_type: Linear solver ("bicgstab", "cg", "gmres", "cr")
            max_iterations: Maximum iterations for iterative solver
            tolerance: Convergence tolerance for iterative solver
            rebuild_matrix_every: Rebuild system matrix every N steps
                                  (1 = every step for nonlinear accuracy,
                                   higher = amortize cost but less accurate)
        """
        super().__init__(model)
        
        self.mass = mass
        self.Minv = 1.0 / self.mass
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.solver_type = solver_type
        self.preconditioner_type = preconditioner_type
        self.rebuild_matrix_every = rebuild_matrix_every
        self._step_count = 0
        
        # Count total blocks needed
        spring_blocks = model.spring_count * 4  # 4 blocks per spring
        
        # Check if model has FEM triangles
        self.has_fem = hasattr(model, 'tri_count') and model.tri_count > 0
        fem_blocks = model.tri_count * 9 if self.has_fem else 0  # 9 blocks per triangle
        
        total_blocks = spring_blocks + fem_blocks
        
        # Pre-allocate arrays for BSR matrix construction
        # Spring blocks
        self.spring_rows = wp.zeros(spring_blocks, dtype=wp.int32, device=model.device)
        self.spring_cols = wp.zeros(spring_blocks, dtype=wp.int32, device=model.device)
        self.spring_values = wp.zeros(spring_blocks, dtype=wp.mat22f, device=model.device)
        
        # FEM blocks (if applicable)
        if self.has_fem:
            self.fem_rows = wp.zeros(fem_blocks, dtype=wp.int32, device=model.device)
            self.fem_cols = wp.zeros(fem_blocks, dtype=wp.int32, device=model.device)
            self.fem_values = wp.zeros(fem_blocks, dtype=wp.mat22f, device=model.device)
        
        # Combined arrays for BSR assembly
        self.bsr_rows = wp.zeros(total_blocks, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(total_blocks, dtype=wp.int32, device=model.device)
        self.bsr_values = wp.zeros(total_blocks, dtype=wp.mat22f, device=model.device)
        
        # Initialize BSR system matrix
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat22f,
            device=model.device
        )
        
        # Preconditioner (will be set after matrix build)
        self.M_bsr = None
        
        # Solution vector
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        
        # External forces buffer
        self.external_forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        
        # Store dt for matrix rebuild check
        self._current_dt = dt
    
    def build_system_matrix(self, model, state, dt: float):
        """
        Build the complete system matrix including spring and FEM stiffness.
        
        A = M - h*D - h²*(K_spring + K_fem)
        
        This needs to be called when:
        - Initial setup
        - Time step changes
        - FEM stiffness changes significantly (large deformation)
        """
        # Build spring blocks
        wp.launch(
            kernel=build_combined_system_matrix_2d,
            dim=model.spring_count,
            inputs=[
                self.spring_rows,
                self.spring_cols,
                self.spring_values,
                model.spring_indices,
                model.spring_stiffness,
                model.spring_damping,
                wp.float32(dt),
                self.mass,
                self.Minv
            ],
            device=model.device
        )
        
        # Build FEM blocks (if applicable)
        if self.has_fem:
            wp.launch(
                kernel=compute_fem_stiffness_blocks_2d,
                dim=model.tri_count,
                inputs=[
                    state.particle_q,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                    self.fem_rows,
                    self.fem_cols,
                    self.fem_values,
                    wp.float32(dt),
                ],
                device=model.device
            )
        
        # Combine spring and FEM blocks into single arrays
        spring_count = model.spring_count * 4
        
        # Copy spring blocks
        wp.copy(self.bsr_rows, self.spring_rows, count=spring_count)
        wp.copy(self.bsr_cols, self.spring_cols, count=spring_count)
        wp.copy(self.bsr_values, self.spring_values, count=spring_count)
        
        # Copy FEM blocks (offset by spring count)
        if self.has_fem:
            fem_count = model.tri_count * 9
            wp.copy(
                self.bsr_rows, self.fem_rows,
                dest_offset=spring_count, count=fem_count
            )
            wp.copy(
                self.bsr_cols, self.fem_cols,
                dest_offset=spring_count, count=fem_count
            )
            wp.copy(
                self.bsr_values, self.fem_values,
                dest_offset=spring_count, count=fem_count
            )
        
        # Assemble BSR matrix
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True
        )
        
        # Update preconditioner
        self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)
        
        self._current_dt = dt
    
    def eval_spring_forces(self, model, state):
        """Evaluate spring forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if model.spring_count > 0:
            eval_spring_forces_2d(model, state, forces, model.spring_strains)
        return forces
    
    def eval_fem_forces(self, model, state):
        """Evaluate FEM forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if self.has_fem:
            eval_triangle_fem_forces_implicit_2d(model, state, forces)
        return forces
    
    def eval_gravity_forces(self, model):
        """Evaluate gravity forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if model.particle_count > 0:
            gravity_force_vec = model.gravity.numpy()[0]
            gravity_force = wp.vec2(gravity_force_vec[0] * self.mass, gravity_force_vec[1] * self.mass)
            
            wp.launch(
                kernel=eval_gravity_2d,
                dim=model.particle_count,
                inputs=[gravity_force],
                outputs=[forces],
                device=model.device
            )
        return forces
    
    def implicit_integration(self, model, state_in, state_out, dt: float):
        """
        Perform implicit integration using iterative linear solver.
        
        Solves: A * dv = f * dt
        Then: v_{n+1} = v_n + dv, x_{n+1} = x_n + v_{n+1} * dt
        """
        # Select solver
        if self.solver_type == "cg":
            solve_fn = cg
        elif self.solver_type == "bicgstab":
            solve_fn = bicgstab
        elif self.solver_type == "gmres":
            solve_fn = gmres
        elif self.solver_type == "cr":
            solve_fn = cr
        else:
            raise ValueError(f"Invalid solver type: {self.solver_type}")
        
        iterations, residual, _ = solve_fn(
            self.A_bsr,
            state_in.particle_f,
            self.dv,
            tol=self.tolerance,
            maxiter=self.max_iterations,
            M=self.M_bsr,
            use_cuda_graph=True
        )
        
        # Update state
        wp.launch(
            kernel=update_state_2d,
            dim=model.particle_count,
            inputs=[
                self.dv,
                dt,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd
            ],
            device=model.device
        )
        
        return iterations, residual
    
    def step(self, state_in, state_out, dt: float, external_forces=None):
        """
        Advance the simulation by one timestep using fully implicit integration.
        
        Args:
            state_in: The input state
            state_out: The output state
            dt: The timestep (in seconds)
            external_forces: Optional external forces (numpy array or wp.array)
        
        Returns:
            state_out: The updated state
        """
        model = self.model
        
        # Rebuild system matrix if needed
        should_rebuild = (
            self._step_count % self.rebuild_matrix_every == 0 or
            self._current_dt != dt or
            self.M_bsr is None
        )
        
        if should_rebuild:
            self.build_system_matrix(model, state_in, dt)
        
        self._step_count += 1
        
        # Handle external forces
        if external_forces is not None:
            if isinstance(external_forces, wp.array):
                wp.copy(self.external_forces, external_forces)
            else:
                temp = wp.array(external_forces, dtype=wp.vec2, device='cpu')
                wp.copy(self.external_forces, temp)
        else:
            self.external_forces.zero_()
        
        # Zero force accumulator
        state_in.particle_f.zero_()
        
        # Evaluate all forces
        spring_forces = self.eval_spring_forces(model, state_in)
        fem_forces = self.eval_fem_forces(model, state_in)
        gravity_forces = self.eval_gravity_forces(model)
        
        # Combine forces (scaled by dt for RHS)
        state_in.particle_f = dt * (
            spring_forces + 
            fem_forces + 
            gravity_forces + 
            self.external_forces
        )
        
        # Solve and update
        self.implicit_integration(model, state_in, state_out, dt)
        
        # Apply boundary conditions
        wp.launch(
            kernel=apply_boundary_2d,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                state_out.particle_qd,
                model.boxsize
            ],
            device=model.device
        )
        
        # Update strains
        self._update_and_normalize_strains()
        
        return state_out

