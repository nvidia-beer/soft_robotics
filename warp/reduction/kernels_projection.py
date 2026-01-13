# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Warp GPU Kernels for Projection Operations

Implements GPU-accelerated projection between full and reduced spaces:
- project_to_reduced: q_r = V^T @ (x - x0)
- project_to_full: x = V @ q_r + x0
- project_forces: f_r = V^T @ f

These kernels are optimized for the specific case of 2D soft body simulation
where the DOF count is moderate (100s-1000s) but real-time performance matters.
"""

import warp as wp
import numpy as np


# ============================================================================
# GPU Kernels for Projection
# ============================================================================

@wp.kernel
def _project_to_reduced_kernel(
    positions: wp.array(dtype=wp.vec2),      # Full positions (n_particles,)
    rest_positions: wp.array(dtype=wp.vec2), # Rest positions (n_particles,)
    modes: wp.array2d(dtype=float),          # Modes matrix (n_dof, n_modes)
    q_reduced: wp.array(dtype=float),        # Output: reduced coords (n_modes,)
    n_particles: int,
    n_modes: int,
):
    """
    Project full positions to reduced coordinates.
    
    q_r[mode] = Σᵢ V[i, mode] * (x[i] - x0[i])
    
    Each thread computes one reduced coordinate.
    """
    mode_idx = wp.tid()
    
    if mode_idx >= n_modes:
        return
    
    # Compute dot product: V[:,mode]^T @ (x - x0)
    result = float(0.0)
    
    for i in range(n_particles):
        # Position displacement
        dx = positions[i][0] - rest_positions[i][0]
        dy = positions[i][1] - rest_positions[i][1]
        
        # Accumulate: V[2*i, mode] * dx + V[2*i+1, mode] * dy
        result += modes[2 * i, mode_idx] * dx
        result += modes[2 * i + 1, mode_idx] * dy
    
    q_reduced[mode_idx] = result


@wp.kernel
def _project_to_full_kernel(
    q_reduced: wp.array(dtype=float),        # Reduced coords (n_modes,)
    rest_positions: wp.array(dtype=wp.vec2), # Rest positions (n_particles,)
    modes: wp.array2d(dtype=float),          # Modes matrix (n_dof, n_modes)
    positions: wp.array(dtype=wp.vec2),      # Output: full positions (n_particles,)
    n_particles: int,
    n_modes: int,
):
    """
    Reconstruct full positions from reduced coordinates.
    
    x[i] = Σⱼ V[i, j] * q_r[j] + x0[i]
    
    Each thread computes one particle position.
    """
    particle_idx = wp.tid()
    
    if particle_idx >= n_particles:
        return
    
    # Compute: V[particle,:] @ q_r
    dx = float(0.0)
    dy = float(0.0)
    
    dof_x = 2 * particle_idx
    dof_y = 2 * particle_idx + 1
    
    for j in range(n_modes):
        q_j = q_reduced[j]
        dx += modes[dof_x, j] * q_j
        dy += modes[dof_y, j] * q_j
    
    # Add rest position
    positions[particle_idx] = wp.vec2(
        dx + rest_positions[particle_idx][0],
        dy + rest_positions[particle_idx][1]
    )


@wp.kernel
def _project_forces_to_reduced_kernel(
    forces: wp.array(dtype=wp.vec2),    # Full forces (n_particles,)
    modes: wp.array2d(dtype=float),     # Modes matrix (n_dof, n_modes)
    f_reduced: wp.array(dtype=float),   # Output: reduced forces (n_modes,)
    n_particles: int,
    n_modes: int,
):
    """
    Project full forces to reduced space.
    
    f_r[mode] = Σᵢ V[i, mode] * f[i]
    
    Each thread computes one reduced force component.
    """
    mode_idx = wp.tid()
    
    if mode_idx >= n_modes:
        return
    
    # Compute dot product: V[:,mode]^T @ f
    result = float(0.0)
    
    for i in range(n_particles):
        fx = forces[i][0]
        fy = forces[i][1]
        
        result += modes[2 * i, mode_idx] * fx
        result += modes[2 * i + 1, mode_idx] * fy
    
    f_reduced[mode_idx] = result


@wp.kernel
def _project_velocity_to_reduced_kernel(
    velocities: wp.array(dtype=wp.vec2),  # Full velocities (n_particles,)
    modes: wp.array2d(dtype=float),       # Modes matrix (n_dof, n_modes)
    v_reduced: wp.array(dtype=float),     # Output: reduced velocities (n_modes,)
    n_particles: int,
    n_modes: int,
):
    """
    Project full velocities to reduced space.
    
    v_r[mode] = Σᵢ V[i, mode] * v[i]
    """
    mode_idx = wp.tid()
    
    if mode_idx >= n_modes:
        return
    
    result = float(0.0)
    
    for i in range(n_particles):
        vx = velocities[i][0]
        vy = velocities[i][1]
        
        result += modes[2 * i, mode_idx] * vx
        result += modes[2 * i + 1, mode_idx] * vy
    
    v_reduced[mode_idx] = result


@wp.kernel
def _project_velocity_to_full_kernel(
    v_reduced: wp.array(dtype=float),      # Reduced velocities (n_modes,)
    modes: wp.array2d(dtype=float),        # Modes matrix (n_dof, n_modes)
    velocities: wp.array(dtype=wp.vec2),   # Output: full velocities (n_particles,)
    n_particles: int,
    n_modes: int,
):
    """
    Reconstruct full velocities from reduced coordinates.
    
    v[i] = Σⱼ V[i, j] * v_r[j]
    """
    particle_idx = wp.tid()
    
    if particle_idx >= n_particles:
        return
    
    vx = float(0.0)
    vy = float(0.0)
    
    dof_x = 2 * particle_idx
    dof_y = 2 * particle_idx + 1
    
    for j in range(n_modes):
        v_j = v_reduced[j]
        vx += modes[dof_x, j] * v_j
        vy += modes[dof_y, j] * v_j
    
    velocities[particle_idx] = wp.vec2(vx, vy)


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def project_to_reduced_space(
    positions: wp.array,
    rest_positions: wp.array,
    modes: wp.array,
    q_reduced: wp.array,
    n_particles: int,
    n_modes: int,
    device: str = 'cuda',
):
    """
    Project full positions to reduced coordinates.
    
    Args:
        positions: Full particle positions, wp.array(dtype=wp.vec2), shape (n_particles,)
        rest_positions: Rest configuration, wp.array(dtype=wp.vec2), shape (n_particles,)
        modes: Basis matrix, wp.array2d(dtype=float), shape (n_dof, n_modes)
        q_reduced: Output reduced coords, wp.array(dtype=float), shape (n_modes,)
        n_particles: Number of particles
        n_modes: Number of modes
        device: Warp device
    """
    wp.launch(
        kernel=_project_to_reduced_kernel,
        dim=n_modes,
        inputs=[positions, rest_positions, modes, q_reduced, n_particles, n_modes],
        device=device,
    )


def project_to_full_space(
    q_reduced: wp.array,
    rest_positions: wp.array,
    modes: wp.array,
    positions: wp.array,
    n_particles: int,
    n_modes: int,
    device: str = 'cuda',
):
    """
    Reconstruct full positions from reduced coordinates.
    
    Args:
        q_reduced: Reduced coordinates, wp.array(dtype=float), shape (n_modes,)
        rest_positions: Rest configuration, wp.array(dtype=wp.vec2), shape (n_particles,)
        modes: Basis matrix, wp.array2d(dtype=float), shape (n_dof, n_modes)
        positions: Output full positions, wp.array(dtype=wp.vec2), shape (n_particles,)
        n_particles: Number of particles
        n_modes: Number of modes
        device: Warp device
    """
    wp.launch(
        kernel=_project_to_full_kernel,
        dim=n_particles,
        inputs=[q_reduced, rest_positions, modes, positions, n_particles, n_modes],
        device=device,
    )


def project_forces_to_reduced(
    forces: wp.array,
    modes: wp.array,
    f_reduced: wp.array,
    n_particles: int,
    n_modes: int,
    device: str = 'cuda',
):
    """
    Project full forces to reduced space.
    
    Args:
        forces: Full forces, wp.array(dtype=wp.vec2), shape (n_particles,)
        modes: Basis matrix, wp.array2d(dtype=float), shape (n_dof, n_modes)
        f_reduced: Output reduced forces, wp.array(dtype=float), shape (n_modes,)
        n_particles: Number of particles
        n_modes: Number of modes
        device: Warp device
    """
    wp.launch(
        kernel=_project_forces_to_reduced_kernel,
        dim=n_modes,
        inputs=[forces, modes, f_reduced, n_particles, n_modes],
        device=device,
    )


def project_velocities_to_reduced(
    velocities: wp.array,
    modes: wp.array,
    v_reduced: wp.array,
    n_particles: int,
    n_modes: int,
    device: str = 'cuda',
):
    """Project full velocities to reduced space."""
    wp.launch(
        kernel=_project_velocity_to_reduced_kernel,
        dim=n_modes,
        inputs=[velocities, modes, v_reduced, n_particles, n_modes],
        device=device,
    )


def project_velocities_to_full(
    v_reduced: wp.array,
    modes: wp.array,
    velocities: wp.array,
    n_particles: int,
    n_modes: int,
    device: str = 'cuda',
):
    """Reconstruct full velocities from reduced coordinates."""
    wp.launch(
        kernel=_project_velocity_to_full_kernel,
        dim=n_particles,
        inputs=[v_reduced, modes, velocities, n_particles, n_modes],
        device=device,
    )

