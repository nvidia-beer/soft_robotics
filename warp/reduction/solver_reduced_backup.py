# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Reduced-Order Solver for 2D Soft Body Simulation

Performs simulation in the reduced space (r dimensions) instead of 
full space (N dimensions), providing significant speedup:
- Linear solve: O(r³) instead of O(N^1.5)
- Force projection: O(Nr) instead of O(N²)
- With hyperreduction: O(|RID| * r) force evaluation

Based on:
- Goury & Duriez, "Fast, generic and reliable control and simulation of soft robots"
- SOFA ModelOrderReduction plugin

Mathematical Formulation:
------------------------
Full system: (M - h*D - h²*K) Δv = h*f
Reduced system: (M_r - h*D_r - h²*K_r) Δv_r = h*f_r

where:
    M_r = V^T M V (reduced mass matrix)
    K_r = V^T K V (reduced stiffness matrix)
    f_r = V^T f   (reduced forces)
    
Reconstruction:
    Δv = V Δv_r
    v_{n+1} = v_n + Δv
    x_{n+1} = x_n + h*v_{n+1}
"""

import numpy as np
import warp as wp
from typing import Optional, Any
from scipy import linalg as scipy_linalg

from .kernels_projection import (
    project_to_reduced_space,
    project_to_full_space,
    project_forces_to_reduced,
    project_velocities_to_reduced,
    project_velocities_to_full,
)


# GPU kernels for efficient force accumulation
@wp.kernel
def add_gravity_kernel(
    forces: wp.array(dtype=wp.vec2),
    gravity: wp.array(dtype=wp.vec2),
    mass: float,
):
    """Add gravity force to all particles on GPU."""
    i = wp.tid()
    g = gravity[0]
    forces[i] = forces[i] + wp.vec2(g[0] * mass, g[1] * mass)


@wp.kernel
def add_external_forces_kernel(
    forces: wp.array(dtype=wp.vec2),
    external: wp.array(dtype=wp.vec2),
):
    """Add external forces on GPU."""
    i = wp.tid()
    forces[i] = forces[i] + external[i]


class SolverReduced:
    """
    Reduced-order implicit solver for 2D soft body simulation.
    
    Uses a pre-computed POD basis to solve in reduced space.
    Optionally uses hyperreduction (ECSW) for faster force evaluation.
    
    Example:
        >>> from warp.reduction import SolverReduced, load_reduced_model
        >>> 
        >>> # Load reduced basis
        >>> reduced_data = load_reduced_model('reduced_basis/')
        >>> 
        >>> # Create solver
        >>> solver = SolverReduced(model, reduced_data)
        >>> 
        >>> # Simulation loop
        >>> state_in = model.state()
        >>> state_out = model.state()
        >>> 
        >>> for step in range(1000):
        >>>     solver.step(state_in, state_out, dt=0.01)
        >>>     state_in, state_out = state_out, state_in
    """
    
    def __init__(
        self,
        model: Any,
        reduced_data: dict,
        mass: float = 1.0,
        use_hyperreduction: bool = True,
        tolerance: float = 1e-6,
    ):
        """
        Initialize reduced solver.
        
        Args:
            model: Warp Model object (full-order model)
            reduced_data: Dictionary with reduced basis data:
                - 'modes': Basis matrix V (n_dof, n_modes)
                - 'rest_position': Rest configuration (n_particles, 2)
                - 'n_modes': Number of modes
                - 'n_particles': Number of particles
                - 'rid': (Optional) Reduced Integration Domain indices
                - 'weights': (Optional) ECSW weights
            mass: Particle mass (uniform)
            use_hyperreduction: Whether to use ECSW for force evaluation
            tolerance: Linear solver tolerance
        """
        self.model = model
        self.mass = mass
        self.use_hyperreduction = use_hyperreduction
        self.tolerance = tolerance
        
        # Extract reduced basis data
        modes_np = reduced_data['modes']
        rest_pos_np = reduced_data['rest_position']
        self.n_modes = reduced_data['n_modes']
        self.n_particles = reduced_data['n_particles']
        self.n_dof = 2 * self.n_particles
        
        # Validate dimensions
        if modes_np.shape[0] != self.n_dof:
            raise ValueError(f"Modes shape mismatch: {modes_np.shape[0]} != {self.n_dof}")
        if modes_np.shape[1] != self.n_modes:
            raise ValueError(f"Modes shape mismatch: {modes_np.shape[1]} != {self.n_modes}")
        
        # Convert to warp arrays
        device = model.device
        self.device = device
        
        # Modes matrix (n_dof, n_modes)
        self.modes = wp.array2d(modes_np.astype(np.float32), dtype=float, device=device)
        
        # Rest positions (n_particles, 2) -> (n_particles,) of vec2
        if rest_pos_np.ndim == 1:
            rest_pos_np = rest_pos_np.reshape(-1, 2)
        self.rest_positions = wp.array(rest_pos_np.astype(np.float32), dtype=wp.vec2, device=device)
        
        # Hyperreduction data (optional)
        self.rid = None
        self.weights = None
        if use_hyperreduction and 'rid' in reduced_data:
            self.rid = reduced_data['rid']
            self.weights = reduced_data['weights']
            print(f"  Using hyperreduction with {len(self.rid)} elements")
        
        # Pre-compute reduced mass matrix: M_r = V^T M V
        # IMPORTANT: Full solver accumulates mass PER SPRING, not per particle!
        # Each spring adds 'mass' to both endpoints' diagonals.
        self.M_r = self._compute_reduced_mass()
        
        # Pre-compute reduced stiffness matrix (if model has constant stiffness)
        # For nonlinear systems, this needs to be updated each step
        self.K_r = self._compute_reduced_stiffness()
        
        # Pre-compute reduced damping matrix: D_r = V^T D V
        # Note: In the system matrix, damping is scaled by 1/mass (Minv)
        self.D_r = self._compute_reduced_damping()
        
        # Allocate reduced space arrays
        self.q_r = wp.zeros(self.n_modes, dtype=float, device=device)
        self.v_r = wp.zeros(self.n_modes, dtype=float, device=device)
        self.f_r = wp.zeros(self.n_modes, dtype=float, device=device)
        self.dv_r = wp.zeros(self.n_modes, dtype=float, device=device)
        
        # Allocate full-space force buffer (for force evaluation)
        self.f_full = wp.zeros(self.n_particles, dtype=wp.vec2, device=device)
        
        # System matrix (M_r - h²*K_r) - computed on first step
        self._A_r = None
        self._lu_piv = None  # LU factorization pivot indices
        self._current_dt = None
        
        # Debug flag (set to 0 to disable debug output)
        self._debug_step = 0
        self._debug_interval = 0  # 0 = disabled, >0 = print every N steps
        
        # Step counter for strain normalization updates
        self._step_count = 0
        
        print(f"SolverReduced initialized:")
        print(f"  Full DOF: {self.n_dof}")
        print(f"  Reduced DOF: {self.n_modes}")
        print(f"  Compression: {self.n_dof / self.n_modes:.1f}x")
        print(f"  Linear solve speedup: ~{(self.n_dof / self.n_modes) ** 2:.0f}x")
        print(f"  M_r diagonal range: [{np.min(np.diag(self.M_r)):.2f}, {np.max(np.diag(self.M_r)):.2f}]")
        print(f"  K_r diagonal range: [{np.min(np.diag(self.K_r)):.2f}, {np.max(np.diag(self.K_r)):.2f}]")
        print(f"  D_r diagonal range: [{np.min(np.diag(self.D_r)):.2f}, {np.max(np.diag(self.D_r)):.2f}]")
    
    def _compute_reduced_mass(self) -> np.ndarray:
        """
        Compute reduced mass matrix M_r = V^T M V.
        
        IMPORTANT: The full solver (build_system_matrix_sparse_2d) accumulates
        mass PER SPRING, not per particle. Each spring adds 'mass' to both
        endpoint particles' diagonal blocks.
        
        This means M[i,i] = n_springs_attached_to_i * mass * I
        """
        model = self.model
        modes = self.modes.numpy()  # (n_dof, n_modes)
        n_dof = self.n_dof
        n_modes = self.n_modes
        
        # Build full mass matrix (diagonal, but with mass per spring)
        M_diag = np.zeros(n_dof, dtype=np.float32)
        
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            spring_indices = model.spring_indices.numpy()
            
            for s in range(model.spring_count):
                i = spring_indices[s * 2]
                j = spring_indices[s * 2 + 1]
                
                # Each spring adds mass to both endpoints
                M_diag[i*2] += self.mass
                M_diag[i*2+1] += self.mass
                M_diag[j*2] += self.mass
                M_diag[j*2+1] += self.mass
        
        # If no springs, fall back to uniform mass per particle
        if np.sum(M_diag) < 1e-8:
            M_diag = np.ones(n_dof, dtype=np.float32) * self.mass
        
        # Project: M_r = V^T @ diag(M_diag) @ V
        # = sum_i M_diag[i] * V[i,:].T @ V[i,:]
        M_r = np.zeros((n_modes, n_modes), dtype=np.float32)
        for i in range(n_dof):
            M_r += M_diag[i] * np.outer(modes[i, :], modes[i, :])
        
        # Ensure symmetry
        M_r = 0.5 * (M_r + M_r.T)
        
        # Add small regularization
        M_r += np.eye(n_modes, dtype=np.float32) * 1e-6
        
        print(f"  Built mass matrix with {model.spring_count} springs")
        
        return M_r.astype(np.float32)
    
    def _build_full_stiffness_matrix(self) -> np.ndarray:
        """
        Build the full stiffness matrix K in dense form.
        K is (n_dof, n_dof) where n_dof = n_particles * 2.
        
        IMPORTANT: Only includes SPRING stiffness, NOT FEM!
        This matches the full solver (SolverImplicit) which only has
        spring stiffness in the system matrix. FEM forces are evaluated
        explicitly and added to the RHS.
        """
        model = self.model
        n_dof = self.n_dof
        
        K = np.zeros((n_dof, n_dof), dtype=np.float32)
        
        # Spring stiffness contribution ONLY (matches full solver)
        # IMPORTANT: Full solver uses ISOTROPIC spring model (k * I), not directional (k * d⊗d)
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            spring_indices = model.spring_indices.numpy()
            spring_stiffness = model.spring_stiffness.numpy()
            
            for s in range(model.spring_count):
                i = spring_indices[s * 2]
                j = spring_indices[s * 2 + 1]
                k = spring_stiffness[s]
                
                # Isotropic spring model: K_block = k * I (2x2 identity)
                # This matches build_system_matrix_sparse_2d kernel
                # K[ii] += k*I, K[jj] += k*I
                # K[ij] -= k*I, K[ji] -= k*I
                for a in range(2):
                    K[i*2+a, i*2+a] += k
                    K[j*2+a, j*2+a] += k
                    K[i*2+a, j*2+a] -= k
                    K[j*2+a, i*2+a] -= k
        
        # NOTE: FEM stiffness is NOT included here!
        # FEM forces are evaluated explicitly in the force evaluation step,
        # just like the full solver does.
        
        return K
    
    def _build_full_damping_matrix(self) -> np.ndarray:
        """
        Build the full damping matrix D in dense form.
        
        IMPORTANT: Only includes SPRING damping, NOT FEM!
        This matches the full solver (SolverImplicit).
        
        Uses ISOTROPIC damping model (c * I) to match build_system_matrix_sparse_2d.
        """
        model = self.model
        n_dof = self.n_dof
        
        D = np.zeros((n_dof, n_dof), dtype=np.float32)
        
        # Spring damping contribution ONLY (matches full solver)
        # IMPORTANT: Full solver uses ISOTROPIC damping model (c * I), not directional
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            spring_indices = model.spring_indices.numpy()
            spring_damping = model.spring_damping.numpy()
            
            for s in range(model.spring_count):
                i = spring_indices[s * 2]
                j = spring_indices[s * 2 + 1]
                c = spring_damping[s]
                
                # Isotropic damping model: D_block = c * I (2x2 identity)
                # This matches build_system_matrix_sparse_2d kernel
                # D[ii] += c*I, D[jj] += c*I
                # D[ij] -= c*I, D[ji] -= c*I
                for a in range(2):
                    D[i*2+a, i*2+a] += c
                    D[j*2+a, j*2+a] += c
                    D[i*2+a, j*2+a] -= c
                    D[j*2+a, i*2+a] -= c
        
        # NOTE: FEM damping is NOT included here!
        # This matches the full solver which only has spring damping in the matrix.
        
        return D
    
    def _compute_reduced_stiffness(self) -> np.ndarray:
        """
        Compute reduced stiffness matrix K_r = V^T K V.
        
        Builds the full stiffness matrix and projects it accurately.
        """
        modes = self.modes.numpy()  # (n_dof, n_modes)
        
        # Build full stiffness matrix
        K_full = self._build_full_stiffness_matrix()
        
        # Project: K_r = V^T @ K @ V
        K_r = modes.T @ K_full @ modes
        
        # Ensure symmetry
        K_r = 0.5 * (K_r + K_r.T)
        
        # Add small regularization
        K_r += np.eye(self.n_modes, dtype=np.float32) * 1e-6
        
        print(f"  Built full K matrix ({self.n_dof}x{self.n_dof}) and projected to ({self.n_modes}x{self.n_modes})")
        
        return K_r.astype(np.float32)
    
    def _compute_reduced_damping(self) -> np.ndarray:
        """
        Compute reduced damping matrix D_r = V^T D V.
        
        Builds the full damping matrix and projects it accurately.
        """
        modes = self.modes.numpy()  # (n_dof, n_modes)
        
        # Build full damping matrix
        D_full = self._build_full_damping_matrix()
        
        # Project: D_r = V^T @ D @ V
        D_r = modes.T @ D_full @ modes
        
        # Ensure symmetry
        D_r = 0.5 * (D_r + D_r.T)
        
        # Add small regularization
        D_r += np.eye(self.n_modes, dtype=np.float32) * 1e-6
        
        print(f"  Built full D matrix ({self.n_dof}x{self.n_dof}) and projected to ({self.n_modes}x{self.n_modes})")
        
        return D_r.astype(np.float32)
    
    def _build_system_matrix(self, dt: float) -> np.ndarray:
        """
        Build reduced system matrix matching build_system_matrix_sparse_2d:
        
        A = M + (dt/mass)*D - dt²*K
        
        The full solver uses: diag_coeff = mass + dt*d/mass - dt²*k
        where the damping term has a POSITIVE sign and is scaled by 1/mass (Minv).
        """
        dt2 = dt * dt
        Minv = 1.0 / self.mass
        
        # Match full solver: A = M + (dt * Minv) * D - dt² * K
        A_r = self.M_r + (dt * Minv) * self.D_r - dt2 * self.K_r
        return A_r
    
    def _lu_solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve A_r @ x = rhs using cached LU factorization.
        O(r²) per solve instead of O(r³) for direct solve.
        """
        return scipy_linalg.lu_solve(self._lu_piv, rhs)
    
    def step(
        self,
        state_in: Any,
        state_out: Any,
        dt: float,
        external_forces: Optional[wp.array] = None,
    ):
        """
        Advance simulation by one timestep using reduced-order integration.
        
        Algorithm:
        1. Project current state to reduced space
        2. Evaluate forces in full space (or hyperreduced)
        3. Project forces to reduced space
        4. Solve reduced linear system
        5. Reconstruct full state
        
        Args:
            state_in: Input state (full-order)
            state_out: Output state buffer (full-order)
            dt: Time step
            external_forces: Optional external forces (full-order)
        """
        model = self.model
        
        # Rebuild system matrix and LU factorization if dt changed
        if self._A_r is None or self._current_dt != dt:
            self._A_r = self._build_system_matrix(dt)
            self._lu_piv = scipy_linalg.lu_factor(self._A_r)  # O(r³) once
            self._current_dt = dt
        
        # -----------------------------------------------------------------
        # Step 1: Project current state to reduced space
        # -----------------------------------------------------------------
        project_to_reduced_space(
            positions=state_in.particle_q,
            rest_positions=self.rest_positions,
            modes=self.modes,
            q_reduced=self.q_r,
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        project_velocities_to_reduced(
            velocities=state_in.particle_qd,
            modes=self.modes,
            v_reduced=self.v_r,
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        # -----------------------------------------------------------------
        # Step 1b: Reconstruct BOTH positions AND velocities to ensure
        # they're IN the subspace. This is critical because:
        # - Positions may have been pushed out by SDF collision
        # - Velocities must be consistent with reduced coords for damping forces
        # -----------------------------------------------------------------
        project_to_full_space(
            q_reduced=self.q_r,
            rest_positions=self.rest_positions,
            modes=self.modes,
            positions=state_in.particle_q,  # Overwrite with reconstructed
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        project_velocities_to_full(
            v_reduced=self.v_r,
            modes=self.modes,
            velocities=state_in.particle_qd,  # Overwrite with reconstructed
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        # -----------------------------------------------------------------
        # Step 2: Evaluate forces in full space (on SUBSPACE state)
        # -----------------------------------------------------------------
        self.f_full.zero_()
        
        # Spring forces
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            from solvers.implicit.kernels_fem_2d import eval_spring_forces_2d
            eval_spring_forces_2d(model, state_in, self.f_full)
        
        # FEM forces
        if hasattr(model, 'tri_count') and model.tri_count > 0:
            from solvers.implicit.kernels_fem_2d import eval_triangle_fem_forces_2d
            eval_triangle_fem_forces_2d(model, state_in, self.f_full)
        
        # Gravity - use GPU kernel
        if model.gravity is not None:
            wp.launch(
                kernel=add_gravity_kernel,
                dim=self.n_particles,
                inputs=[self.f_full, model.gravity, self.mass],
                device=self.device
            )
        
        # External forces - use GPU kernel  
        if external_forces is not None:
            wp.launch(
                kernel=add_external_forces_kernel,
                dim=self.n_particles,
                inputs=[self.f_full, external_forces],
                device=self.device
            )
        
        # -----------------------------------------------------------------
        # Step 3: Project forces to reduced space
        # -----------------------------------------------------------------
        project_forces_to_reduced(
            forces=self.f_full,
            modes=self.modes,
            f_reduced=self.f_r,
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        # Debug: measure force projection quality (disabled by default)
        self._debug_step += 1
        if self._debug_interval > 0 and self._debug_step % self._debug_interval == 1:
            wp.synchronize_device(self.device)
            f_full_np = self.f_full.numpy()
            f_r_np_debug = self.f_r.numpy()
            modes_np = self.modes.numpy()
            
            # Reconstruct forces from reduced space
            f_full_flat = f_full_np.flatten()
            f_reconstructed = modes_np @ f_r_np_debug
            
            # Force norms
            f_full_norm = np.linalg.norm(f_full_flat)
            f_recon_norm = np.linalg.norm(f_reconstructed)
            f_lost_norm = np.linalg.norm(f_full_flat - f_reconstructed)
            
            if f_full_norm > 1e-8:
                force_retained = 100 * (1 - f_lost_norm / f_full_norm)
            else:
                force_retained = 100.0
            
            print(f"  [Step {self._debug_step}] Force: |f|={f_full_norm:.4f}, retained={force_retained:.1f}%")
        
        # -----------------------------------------------------------------
        # Step 4: Solve reduced linear system  
        # -----------------------------------------------------------------
        # A_r * dv_r = dt * f_r
        # Use cached LU factorization (O(r²) per solve instead of O(r³))
        
        # Sync GPU and batch-read all needed arrays (single transfer point)
        wp.synchronize_device(self.device)
        f_r_np = self.f_r.numpy()
        v_r_np = self.v_r.numpy()
        q_r_np = self.q_r.numpy()
        
        # Solve using pre-factorized matrix
        rhs = dt * f_r_np
        dv_r_np = self._lu_solve(rhs)
        
        # Update in place (no new allocations)
        v_r_np += dv_r_np
        q_r_np += dt * v_r_np
        
        # Single batch write back to GPU
        self.v_r.assign(v_r_np)
        self.q_r.assign(q_r_np)
        
        # -----------------------------------------------------------------
        # Step 5: Reconstruct full state
        # -----------------------------------------------------------------
        project_to_full_space(
            q_reduced=self.q_r,
            rest_positions=self.rest_positions,
            modes=self.modes,
            positions=state_out.particle_q,
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        project_velocities_to_full(
            v_reduced=self.v_r,
            modes=self.modes,
            velocities=state_out.particle_qd,
            n_particles=self.n_particles,
            n_modes=self.n_modes,
            device=self.device,
        )
        
        # Apply boundary conditions (optional - could also do in reduced space)
        from solvers.implicit.kernels_fem_2d import apply_boundary_2d
        wp.launch(
            kernel=apply_boundary_2d,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                state_out.particle_qd,
                model.boxsize
            ],
            device=self.device
        )
        
        # -----------------------------------------------------------------
        # Step 6: Re-evaluate strains on OUTPUT state for correct visualization
        # -----------------------------------------------------------------
        # The strains computed in Step 2 are based on state_in (previous positions).
        # We need to update them based on state_out (current positions) for display.
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            from solvers.implicit.kernels_fem_2d import eval_spring_forces_2d
            # Create dummy force array (we just want strain updates, not forces)
            dummy_forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
            eval_spring_forces_2d(model, state_out, dummy_forces)
        
        if hasattr(model, 'tri_count') and model.tri_count > 0:
            from solvers.implicit.kernels_fem_2d import eval_triangle_fem_forces_2d
            dummy_forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
            eval_triangle_fem_forces_2d(model, state_out, dummy_forces)
        
        # -----------------------------------------------------------------
        # Step 7: Update strain normalization scales (adaptive visualization)
        # -----------------------------------------------------------------
        # This is CRITICAL - without it, strains appear saturated (all red/blue)
        self._step_count += 1
        if self._step_count % 10 == 0:
            self._update_strain_scales()
        
        return state_out
    
    def _update_strain_scales(self):
        """Update adaptive strain normalization using 95th percentile with EMA."""
        model = self.model
        ema_alpha = 0.1
        
        # Spring strains
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            strains_np = model.spring_strains.numpy()
            abs_strains = np.abs(strains_np)
            if len(abs_strains) > 0:
                percentile_95 = np.percentile(abs_strains, 95)
                if percentile_95 < 1e-8:
                    percentile_95 = 0.01
                current_scale = model.spring_strain_scale.numpy()[0]
                new_scale = ema_alpha * percentile_95 + (1 - ema_alpha) * current_scale
                model.spring_strain_scale.assign([new_scale])
        
        # FEM strains
        if hasattr(model, 'tri_count') and model.tri_count > 0:
            strains_np = model.tri_strains.numpy()
            abs_strains = np.abs(strains_np)
            if len(abs_strains) > 0:
                percentile_95 = np.percentile(abs_strains, 95)
                if percentile_95 < 1e-8:
                    percentile_95 = 0.01
                current_scale = model.fem_strain_scale.numpy()[0]
                new_scale = ema_alpha * percentile_95 + (1 - ema_alpha) * current_scale
                model.fem_strain_scale.assign([new_scale])
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return {
            'n_full': self.n_dof,
            'n_reduced': self.n_modes,
            'compression_ratio': self.n_dof / self.n_modes,
            'linear_solve_speedup': (self.n_dof / self.n_modes) ** 2,
            'hyperreduction': self.use_hyperreduction,
            'rid_size': len(self.rid) if self.rid is not None else None,
        }

