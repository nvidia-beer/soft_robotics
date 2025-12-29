# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Implicit solver for 2D spring-mass systems with FEM support
# Adapted from soft/solver_soft.py (3D tetrahedral -> 2D triangular)

import numpy as np
import warp as wp
from warp.optim.linear import cg, preconditioner, bicgstab, gmres, cr
from warp.sparse import BsrMatrix, bsr_zeros, bsr_set_from_triplets

from ..solver import SolverBase
from .kernels_fem_2d import (
    build_system_matrix_sparse_2d,
    eval_spring_forces_2d,
    eval_triangle_fem_forces_2d,
    eval_gravity_2d,
    update_state_2d,
    apply_boundary_2d,
    apply_boundary_with_friction_2d,
)


class SolverImplicit(SolverBase):
    """
    Implicit solver for 2D spring-mass systems using sparse matrix techniques.
    
    Features:
        - Unconditionally stable implicit integration
        - Sparse BSR (Block Sparse Row) matrix representation (2x2 blocks)
        - Iterative linear solvers (BiCGStab, CG, GMRES, CR)
        - Optional triangular FEM elements for continuum mechanics
        - Preconditioned conjugate gradient methods
    
    This is a 2D adaptation of the 3D soft body solver from soft/solver_soft.py.
    The key differences:
        - Uses wp.vec2 instead of wp.vec3
        - Uses wp.mat22f blocks instead of wp.mat33f
        - Uses triangles instead of tetrahedra for FEM
    
    Algorithm Comparison (from soft/solver_soft.py):
    
        BiCGStab (Bi-Conjugate Gradient Stabilized):
        ✓ Pros: Handles non-symmetric systems, 2-5× faster than CG, smooth convergence
        ✗ Cons: Slightly higher memory, can breakdown rarely
        
        CG (Conjugate Gradient):
        ✓ Pros: Minimal memory, simple, theoretical convergence in N steps
        ✗ Cons: REQUIRES symmetric positive definite matrices
        
        GMRES (Generalized Minimal Residual):
        ✓ Pros: Excellent with good preconditioners, works with any matrix
        ✗ Cons: High memory usage (grows with iterations)
        
        CR (Conjugate Residual):
        ✓ Pros: Better than CG for ill-conditioned SPD systems
        ✗ Cons: Still requires symmetric positive definite matrices
    
    Example:
        >>> model = Model.from_grid(N=10, spacing=0.2)
        >>> solver = SolverImplicit(model, dt=0.01, solver_type='bicgstab')
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
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        max_iterations: int = 3,
        tolerance: float = 1e-2,
        ratchet_friction: bool = False,
        locomotion_direction: tuple = (1.0, 0.0),
    ):
        """
        Initialize the implicit solver.
        
        Args:
            model: The 2D Model to be simulated
            dt: Time step for integration
            mass: Particle mass (uniform)
            preconditioner_type: Type of preconditioner ("id", "diag", "diag_abs")
            solver_type: Linear solver ("bicgstab", "cg", "gmres", "cr")
            max_iterations: Maximum iterations for iterative solver
            tolerance: Convergence tolerance for iterative solver
            ratchet_friction: Enable direction-aware ratchet friction for locomotion
            locomotion_direction: 2D direction vector (dx, dy) for ratchet friction.
                (1, 0) = right, (-1, 0) = left, (0, 1) = up, etc.
        """
        super().__init__(model)
        
        # Ratchet friction for locomotion
        self.ratchet_friction = ratchet_friction
        self.ground_y = 0.0
        
        # Locomotion direction (normalized) as wp.vec2
        import numpy as np
        d = np.array(locomotion_direction, dtype=float)
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            d = d / norm
        else:
            d = np.array([1.0, 0.0])
        self.direction = wp.vec2(float(d[0]), float(d[1]))
        
        # Disable ratchet for vertical movement (|dy| > |dx|)
        # Ratchet friction only makes sense for horizontal crawling
        if abs(d[1]) > abs(d[0]):
            self.ratchet_friction = False
        
        self.mass = mass
        self.Minv = 1.0 / self.mass
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.solver_type = solver_type
        
        # Pre-allocate arrays for BSR matrix construction (2x2 blocks)
        num_blocks = model.spring_count * 4  # Each edge contributes 4 blocks
        self.bsr_rows = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_values = wp.zeros(num_blocks, dtype=wp.mat22f, device=model.device)
        
        # Initialize BSR system matrix (2x2 blocks for 2D)
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat22f,
            device=model.device
        )
        
        # Build and assemble the system matrix
        self.initialize_system_matrix_sparse(model, dt)
        
        # Preconditioner
        # - "diag": Diagonal (Jacobi) preconditioner
        # - "diag_abs": Jacobi using absolute value of diagonal
        # - "id": Identity (no preconditioning)
        self.M_bsr = preconditioner(self.A_bsr, ptype=preconditioner_type)
        
        # Solution vector (velocity change)
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        
        # External forces buffer
        self.external_forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
    
    def initialize_system_matrix_sparse(self, model, dt: float):
        """
        Build the sparse system matrix A = M - h*D - h^2*K.
        
        This matrix represents the linearized implicit system and is constant
        for linear springs with fixed stiffness and damping.
        """
        wp.launch(
            kernel=build_system_matrix_sparse_2d,
            dim=model.spring_count,
            inputs=[
                self.bsr_rows,
                self.bsr_cols,
                self.bsr_values,
                model.spring_indices,
                model.spring_stiffness,
                model.spring_damping,
                wp.float32(dt),
                self.mass,
                self.Minv
            ],
            device=model.device
        )
        
        # Assemble BSR matrix from computed blocks
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True
        )
    
    def eval_spring_forces(self, model, state):
        """Evaluate spring forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if model.spring_count > 0:
            eval_spring_forces_2d(model, state, forces, model.spring_strains)
        return forces
    
    def eval_fem_forces(self, model, state):
        """Evaluate FEM forces for triangular elements."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if hasattr(model, 'tri_count') and model.tri_count > 0:
            eval_triangle_fem_forces_2d(model, state, forces)
        return forces
    
    def eval_gravity_forces(self, model):
        """Evaluate gravity forces for all particles."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        if model.particle_count > 0:
            # F = m * g
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
        where A = M - h*D - h^2*K (system matrix)
              dv = v_{n+1} - v_n (velocity change)
              f = total forces
        
        Then updates:
              v_{n+1} = v_n + dv
              x_{n+1} = x_n + v_{n+1} * dt
        """
        # Choose solver based on solver_type
        if self.solver_type == "cg":
            iterations, residual, _ = cg(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=self.tolerance,
                maxiter=self.max_iterations,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "bicgstab":
            iterations, residual, _ = bicgstab(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=self.tolerance,
                maxiter=self.max_iterations,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "gmres":
            iterations, residual, _ = gmres(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=self.tolerance,
                maxiter=self.max_iterations,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "cr":
            iterations, residual, _ = cr(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=self.tolerance,
                maxiter=self.max_iterations,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        else:
            raise ValueError(f"Invalid solver type: {self.solver_type}")
        
        # Update state using computed velocity change
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
    
    def step(self, state_in, state_out, dt: float, external_forces=None):
        """
        Advance the simulation by one timestep using implicit integration.
        
        Args:
            state_in: The input state
            state_out: The output state
            dt: The timestep (in seconds)
            external_forces: Optional external forces (numpy array or wp.array)
        
        Returns:
            state_out: The updated state
        """
        model = self.model
        
        # Handle external forces
        if external_forces is not None:
            if isinstance(external_forces, wp.array):
                wp.copy(self.external_forces, external_forces)
            else:
                # Assume numpy array
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
        
        # Combine forces: F_total = F_spring + F_fem + F_gravity + F_external
        # Scale by dt for right-hand side of linear system
        state_in.particle_f = dt * (
            spring_forces + 
            fem_forces + 
            gravity_forces + 
            self.external_forces
        )
        
        # Solve implicit system and update state
        self.implicit_integration(model, state_in, state_out, dt)
        
        # Apply boundary conditions (with or without ratchet friction)
        if self.ratchet_friction:
            wp.launch(
                kernel=apply_boundary_with_friction_2d,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.boxsize,
                    self.ground_y,
                    self.direction,
                ],
                device=model.device
            )
        else:
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
        
        # Update adaptive scale and normalize strains (base class method)
        self._update_and_normalize_strains()
        
        return state_out

