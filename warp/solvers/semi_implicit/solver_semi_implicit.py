# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Semi-implicit solver for 2D spring-mass systems
# Adapted from newton/newton/_src/solvers/semi_implicit/solver_semi_implicit.py

import warp as wp

from ..solver import SolverBase
from .kernels_particle import (
    apply_boundary_2d,
    apply_boundary_with_friction_2d,
    eval_spring_forces_2d,
    finalize_velocity_2d,
    integrate_particles_2d,
)


class SolverSemiImplicit(SolverBase):
    """
    A semi-implicit integrator using symplectic Euler for 2D spring-mass systems.
    
    Semi-implicit time integration preserves energy but is not unconditionally stable.
    Requires a time-step small enough to support the stiffness and damping forces.
    
    Adapted from newton.solvers.SolverSemiImplicit for 2D systems.
    
    Example:
        >>> model = GridModel(rows=5, cols=5, spacing=0.25)
        >>> solver = SolverSemiImplicit(model)
        >>> state_in = model.state()
        >>> state_out = model.state()
        >>>
        >>> for i in range(100):
        >>>     solver.step(state_in, state_out, dt=0.01)
        >>>     state_in, state_out = state_out, state_in
    """
    
    def __init__(self, model, ratchet_friction: bool = False, locomotion_direction: tuple = (1.0, 0.0)):
        """
        Initialize the semi-implicit solver.
        
        Args:
            model: The 2D Model to be simulated
            ratchet_friction: Enable direction-aware ratchet friction for locomotion.
                False = reflective boundaries only (default)
                True = ratchet friction (slides forward, grips backward)
            locomotion_direction: 2D direction vector (dx, dy) for ratchet friction.
                (1, 0) = right, (-1, 0) = left, (0, 1) = up, etc.
        """
        super().__init__(model)
        
        # Allocate external force buffer
        self.external_forces = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
        
        # Ratchet friction for locomotion
        self.ratchet_friction = ratchet_friction
        self.ground_y = 0.0  # Ground level
        
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
    
    def step(self, state_in, state_out, dt: float, external_forces=None):
        """
        Advance the simulation by one timestep.
        
        Args:
            state_in: The input state
            state_out: The output state
            dt: The timestep (in seconds)
            external_forces: Optional external forces to apply (numpy array or wp.array)
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
        
        # Evaluate spring forces at time n
        eval_spring_forces_2d(model, state_in, state_in.particle_f)
        
        # Get gravity vector value
        gravity_vec = model.gravity.numpy()[0]
        gravity_wp = wp.vec2(gravity_vec[0], gravity_vec[1])
        
        # Integrate particles (half-kick and drift)
        wp.launch(
            kernel=integrate_particles_2d,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                gravity_wp,
                dt,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )
        
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
                device=model.device,
            )
        else:
            wp.launch(
                kernel=apply_boundary_2d,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.boxsize,
                ],
                device=model.device,
            )
        
        # Evaluate spring forces at time n+1
        state_in.particle_f.zero_()
        eval_spring_forces_2d(model, state_out, state_in.particle_f)
        
        # Finalize velocity (second half-kick)
        wp.launch(
            kernel=finalize_velocity_2d,
            dim=model.particle_count,
            inputs=[
                state_out.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                gravity_wp,
                self.external_forces,
                dt,
            ],
            outputs=[state_out.particle_qd],
            device=model.device,
        )
        
        # Update adaptive scale and normalize strains (base class method)
        # Semi-implicit only computes spring forces, not FEM
        self._update_and_normalize_strains(update_fem=False)
        
        return state_out

