# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Vertex Block Descent (VBD) solver for 2D FEM simulations
#
# Based on: Chen et al. 2024, "Vertex Block Descent" (SIGGRAPH)
# https://dl.acm.org/doi/10.1145/3658179
#
# Theory:
# -------
# VBD is an optimization-based implicit time integrator that:
# 1. Formulates implicit Euler as an optimization problem
# 2. Uses block coordinate descent (one block per vertex)
# 3. Parallelizes via graph coloring
#
# Advantages over traditional implicit solvers:
# - Near-linear scaling with mesh size on GPUs
# - No global linear system assembly/solve
# - Natural handling of nonlinear materials

import time
import numpy as np
import warp as wp

from ..solver import SolverBase
from .coloring import graph_coloring_2d, compute_adjacency_from_triangles, compute_vertex_to_triangle_adjacency
from .kernels_vbd_2d import (
    vbd_solve_vertex_2d,
    vbd_position_init_2d,
    compute_triangle_masses_2d,
    compute_masses_from_areas_2d,
    compute_triangle_strains_2d,
    compute_spring_strains_2d,
    apply_boundary_vbd_2d,
)


class SolverVBD(SolverBase):
    """
    Vertex Block Descent solver for 2D FEM spring-mass systems.
    
    VBD minimizes the implicit Euler objective function through
    parallel coordinate descent, using graph coloring to identify
    independent vertex updates.
    
    Key Features:
    - GPU-parallel vertex updates via graph coloring
    - Per-vertex 2x2 linear system solve (no global assembly)
    - Stable Neo-Hookean material model
    - Near-linear scaling with mesh size
    
    Algorithm:
    ----------
    For each timestep:
    1. Initialize positions with explicit Euler prediction
    2. For each iteration until convergence:
       a. For each color group (parallelized):
          - Compute gradient and Hessian from incident triangles
          - Solve: dx = -H⁻¹ * g
          - Update: x += dx
       b. Check convergence: ||dx||_∞ < tolerance
    3. Update velocities: v = (x - x_prev) / dt
    
    Example:
        >>> model = GridModel(rows=10, cols=10, spacing=0.2, with_fem=True)
        >>> solver = SolverVBD(model, dt=0.01, dx_tol=1e-6, max_iter=50)
        >>> state_in = model.state()
        >>> state_out = model.state()
        >>> 
        >>> for i in range(1000):
        >>>     solver.step(state_in, state_out, dt=0.01)
        >>>     state_in, state_out = state_out, state_in
    
    References:
    -----------
    Chen et al. 2024, "Vertex Block Descent" (SIGGRAPH)
    Smith et al. 2018, "Stable Neo-Hookean Flesh Simulation"
    """
    
    def __init__(
        self,
        model,
        dt: float = 0.05,
        dx_tol: float = 1e-4,  # Looser tolerance for speed
        max_iter: int = 10,    # Fewer iterations (VBD converges fast)
        damping_coefficient: float = 1.0,
        density: float = 1.0,
        convergence_check_interval: int = 2,  # Check every N iterations
    ):
        """
        Initialize the VBD solver.
        
        Args:
            model: The 2D Model with FEM triangles
            dt: Default time step (can be overridden in step())
            dx_tol: Convergence tolerance for position updates
            max_iter: Maximum VBD iterations per timestep
            damping_coefficient: Damping scale factor (0 = no damping, 1 = standard)
            density: Material density for mass computation
        """
        super().__init__(model)
        
        self.dt = dt
        self.dx_tol = dx_tol
        self.max_iter = max_iter
        self.damping_coefficient = damping_coefficient
        self.density = density
        self.convergence_check_interval = convergence_check_interval
        
        # Cache gravity to avoid repeated GPU->CPU transfers
        self._cached_gravity = None
        
        # Check for FEM triangles
        if not hasattr(model, 'tri_count') or model.tri_count == 0:
            raise ValueError("VBD solver requires FEM triangles. Use with_fem=True in GridModel()")
        
        n_vertices = model.particle_count
        n_triangles = model.tri_count
        
        print(f"Initializing VBD solver for {n_vertices} vertices, {n_triangles} triangles")
        
        # Get triangle indices for graph construction
        tri_indices_np = model.tri_indices.numpy()
        
        # ====================================================================
        # Graph Coloring for Parallel Updates
        # ====================================================================
        start_time = time.time()
        
        # Build vertex adjacency from triangles
        adjacency, max_degree = compute_adjacency_from_triangles(tri_indices_np, n_vertices)
        
        # Perform graph coloring
        coloring, color_groups = graph_coloring_2d(adjacency)
        
        print(f"  Vertex max degree: {max_degree}")
        
        # Convert color groups to fixed-size GPU array
        n_colors = len(color_groups)
        max_group_size = max(len(g) for g in color_groups.values())
        colors_np = np.full((n_colors, max_group_size), -1, dtype=np.int32)
        for c in range(n_colors):
            group = color_groups[c]
            colors_np[c, :len(group)] = group
        
        self.color_groups = wp.array(colors_np, dtype=wp.int32, device=model.device)
        self.n_colors = n_colors
        self.max_group_size = max_group_size
        
        # ====================================================================
        # Vertex-to-Triangle Adjacency
        # ====================================================================
        adj_v2t_np, max_incident = compute_vertex_to_triangle_adjacency(tri_indices_np, n_vertices)
        self.adj_v2t = wp.array(adj_v2t_np, dtype=wp.int32, device=model.device)
        
        print(f"  Max incident triangles per vertex: {max_incident}")
        
        coloring_time = time.time() - start_time
        print(f"  Graph coloring completed in {coloring_time*1000:.2f}ms")
        
        # ====================================================================
        # Pre-allocate Work Buffers
        # ====================================================================
        # Use model's pre-computed triangle areas (from rest configuration)
        # These are computed during model setup and should be correct
        if hasattr(model, 'tri_areas') and model.tri_areas is not None:
            self.tri_areas = model.tri_areas
            print(f"  Using model's pre-computed triangle areas")
        else:
            # Fallback: compute our own
            self.tri_areas = wp.zeros(n_triangles, dtype=wp.float32, device=model.device)
            print(f"  Warning: Computing triangle areas (model.tri_areas not found)")
        
        # Triangle masses (computed from density * area)
        self.tri_masses = wp.zeros(n_triangles, dtype=wp.float32, device=model.device)
        
        # Density array (uniform for now)
        self.densities = wp.full(n_triangles, density, dtype=wp.float32, device=model.device)
        
        # Compute masses from areas
        self._compute_triangle_masses(model)
        
        # Debug: print mass/area stats
        wp.synchronize()
        areas_np = self.tri_areas.numpy()
        masses_np = self.tri_masses.numpy()
        gravity_np = model.gravity.numpy()[0]
        print(f"  Triangle areas: min={areas_np.min():.6f}, max={areas_np.max():.6f}, mean={areas_np.mean():.6f}")
        print(f"  Triangle masses: min={masses_np.min():.6f}, max={masses_np.max():.6f}, sum={masses_np.sum():.6f}")
        print(f"  Gravity: ({gravity_np[0]:.2f}, {gravity_np[1]:.2f})")

        # Active mask (1 = active, 0 = fixed)
        self.active_mask = wp.ones(n_vertices, dtype=wp.int32, device=model.device)
        
        # Work arrays for VBD iteration
        self.old_positions = wp.zeros(n_vertices, dtype=wp.vec2, device=model.device)
        self.old_velocities = wp.zeros(n_vertices, dtype=wp.vec2, device=model.device)
        self.grads = wp.zeros(n_vertices, dtype=wp.vec2, device=model.device)
        self.dxs = wp.zeros(n_vertices, dtype=wp.vec2, device=model.device)
        
        # External forces buffer
        self.external_forces = wp.zeros(n_vertices, dtype=wp.vec2, device=model.device)
        
        print(f"✓ VBD solver initialized ({n_colors} color groups)")
    
    def _compute_triangle_masses(self, model):
        """Compute triangle masses from areas and density."""
        # Use pre-computed areas if available, otherwise compute from positions
        if hasattr(model, 'tri_areas') and model.tri_areas is not None:
            # Use simple mass computation from existing areas
            wp.launch(
                kernel=compute_masses_from_areas_2d,
                dim=model.tri_count,
                inputs=[
                    self.tri_areas,
                    self.densities,
                ],
                outputs=[
                    self.tri_masses,
                ],
                device=model.device
            )
        else:
            # Compute both areas and masses from positions
            wp.launch(
                kernel=compute_triangle_masses_2d,
                dim=model.tri_count,
                inputs=[
                    model.particle_q,
                    model.tri_indices,
                    self.densities,
                ],
                outputs=[
                    self.tri_masses,
                    self.tri_areas,
                ],
                device=model.device
            )
        wp.synchronize()
    
    def set_fixed_vertices(self, vertex_indices):
        """
        Mark vertices as fixed (boundary conditions).
        
        Fixed vertices will not be updated by VBD iterations.
        
        Args:
            vertex_indices: List or array of vertex indices to fix
        """
        active_np = self.active_mask.numpy()
        for idx in vertex_indices:
            if 0 <= idx < len(active_np):
                active_np[idx] = 0
        self.active_mask.assign(wp.array(active_np, dtype=wp.int32, device=self.model.device))
    
    def step(self, state_in, state_out, dt: float, external_forces=None):
        """
        Advance the simulation by one timestep using VBD.
        
        Args:
            state_in: Input state (positions, velocities)
            state_out: Output state (will be updated)
            dt: Timestep size
            external_forces: Optional external forces (numpy or warp array)
        
        Returns:
            state_out: Updated state
        """
        model = self.model
        n_vertices = model.particle_count
        
        # Handle external forces
        if external_forces is not None:
            if isinstance(external_forces, wp.array):
                wp.copy(self.external_forces, external_forces)
            else:
                temp = wp.array(external_forces.reshape(-1, 2).astype(np.float32), 
                               dtype=wp.vec2, device=model.device)
                wp.copy(self.external_forces, temp)
        else:
            self.external_forces.zero_()
        
        # Get gravity (cached to avoid repeated GPU->CPU transfers)
        if self._cached_gravity is None:
            gravity_np = model.gravity.numpy()[0]
            self._cached_gravity = wp.vec2(gravity_np[0], gravity_np[1])
        gravity = self._cached_gravity
        
        # Store previous state
        wp.copy(self.old_positions, state_in.particle_q)
        wp.copy(self.old_velocities, state_in.particle_qd)
        
        # Initialize with explicit Euler prediction
        wp.launch(
            kernel=vbd_position_init_2d,
            dim=n_vertices,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                gravity,
                wp.float32(dt),
                self.active_mask,
            ],
            outputs=[state_out.particle_q],
            device=model.device
        )
        
        # VBD iteration - optimized for speed
        converged = False
        dx_max = 0.0
        
        for iteration in range(self.max_iter):
            # Process all color groups (kernel launches are ordered on same stream)
            for c in range(self.n_colors):
                wp.launch(
                    kernel=vbd_solve_vertex_2d,
                    dim=self.max_group_size,
                    inputs=[
                        state_out.particle_q,
                        self.old_positions,
                        self.old_velocities,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_materials,
                        self.tri_masses,
                        self.tri_areas,
                        gravity,
                        wp.float32(dt),
                        wp.float32(self.damping_coefficient),
                        self.adj_v2t,
                        self.color_groups[c],
                        self.active_mask,
                    ],
                    outputs=[
                        state_out.particle_q,
                        self.grads,
                        self.dxs,
                    ],
                    device=model.device
                )
            
            # Only check convergence periodically (expensive GPU->CPU transfer)
            if (iteration + 1) % self.convergence_check_interval == 0:
                wp.synchronize()
                dxs_np = self.dxs.numpy()
                dx_max = np.abs(dxs_np).max()
                
                if dx_max < self.dx_tol:
                    converged = True
                    break
        
        # Final sync if we didn't check on last iteration
        if not converged:
            wp.synchronize()
        
        # Update velocities (implicit Euler)
        # v_new = (x_new - x_old) / dt
        state_out_q_np = state_out.particle_q.numpy()
        old_q_np = self.old_positions.numpy()
        new_v_np = (state_out_q_np - old_q_np) / dt
        state_out.particle_qd.assign(wp.array(new_v_np.astype(np.float32), 
                                              dtype=wp.vec2, device=model.device))
        
        # Apply boundary conditions
        wp.launch(
            kernel=apply_boundary_vbd_2d,
            dim=n_vertices,
            inputs=[
                state_out.particle_q,
                state_out.particle_qd,
                wp.float32(model.boxsize),
            ],
            device=model.device
        )
        
        # Debug: print center of mass
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            center = state_out_q_np.mean(axis=0)
            print(f"[VBD step {self._step_count}] Center: ({center[0]:.4f}, {center[1]:.4f}), dx_max: {dx_max:.6f}")

        # Update strain visualization
        self._update_strains(model, state_out)
        
        # Update adaptive strain normalization
        has_fem = hasattr(model, 'tri_count') and model.tri_count > 0
        self._update_and_normalize_strains(update_fem=has_fem)
        
        return state_out
    
    def _update_strains(self, model, state):
        """Compute strain values for visualization (both FEM and springs)."""
        # Update FEM triangle strains
        if hasattr(model, 'tri_strains') and model.tri_strains is not None:
            fem_strain_scale = model.fem_strain_scale.numpy()[0]
            
            wp.launch(
                kernel=compute_triangle_strains_2d,
                dim=model.tri_count,
                inputs=[
                    state.particle_q,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                    model.tri_strains,
                    model.tri_strains_normalized,
                    wp.float32(fem_strain_scale),
                ],
                device=model.device
            )
        
        # Update spring strains (for visualization - VBD uses FEM physics, not springs)
        if hasattr(model, 'spring_strains') and model.spring_count > 0:
            spring_strain_scale = model.spring_strain_scale.numpy()[0]
            
            wp.launch(
                kernel=compute_spring_strains_2d,
                dim=model.spring_count,
                inputs=[
                    state.particle_q,
                    model.spring_indices,
                    model.spring_rest_length,
                    model.spring_strains,
                    model.spring_strains_normalized,
                    wp.float32(spring_strain_scale),
                ],
                device=model.device
            )