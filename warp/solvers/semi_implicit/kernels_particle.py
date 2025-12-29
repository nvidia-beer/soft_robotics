# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# 2D spring force kernels for semi-implicit solver
# Adapted from newton/newton/_src/solvers/semi_implicit/kernels_particle.py

import warp as wp


@wp.kernel
def eval_spring_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec2),
    spring_strains: wp.array(dtype=float),  # Output: raw strain
    spring_strains_normalized: wp.array(dtype=float),  # Output: normalized strain [-1, 1]
    strain_scale: float,  # Adaptive normalization scale
):
    """
    Evaluate spring forces using Hooke's Law in 2D.
    
    Adapted from Newton 3D semi-implicit solver (kernels_particle.py:eval_spring).
    Each thread processes one spring connection.
    """
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    if i == -1 or j == -1:
        return

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    
    # Safety check for degenerate springs
    if l < 1e-6:
        spring_strains[tid] = 0.0
        return
    
    l_inv = 1.0 / l

    # Normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)
    
    # Cache raw strain: ε = (L - L₀) / L₀
    raw_strain = c / rest
    spring_strains[tid] = raw_strain
    
    # Normalize strain to [-1, 1] range (branchless for GPU efficiency)
    # Use max to avoid division by zero, then clamp result
    safe_scale = wp.max(strain_scale, 1e-8)
    normalized = wp.clamp(raw_strain / safe_scale, -1.0, 1.0)
    spring_strains_normalized[tid] = normalized

    # Damping based on relative velocity
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def integrate_particles_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec2,
    dt: float,
    x_new: wp.array(dtype=wp.vec2),
    v_new: wp.array(dtype=wp.vec2),
):
    """
    Semi-implicit Euler integration - first half.
    
    Step 1: v_{n+1/2} = v_n + a_n * dt/2  (half kick)
    Step 2: x_{n+1} = x_n + v_{n+1/2} * dt  (drift)
    
    The second half kick is done in finalize_velocity_2d after force evaluation.
    """
    tid = wp.tid()
    
    inv_m = inv_mass[tid]
    
    # Acceleration from forces and gravity
    acc = f[tid] * inv_m + gravity
    
    # Half kick
    v_half = v[tid] + acc * (dt / 2.0)
    
    # Drift
    x_new[tid] = x[tid] + v_half * dt
    v_new[tid] = v_half


@wp.kernel
def finalize_velocity_2d(
    v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec2,
    external_forces: wp.array(dtype=wp.vec2),
    dt: float,
    v_new: wp.array(dtype=wp.vec2),
):
    """
    Finalize velocity integration (second half-kick).
    
    v_{n+1} = v_{n+1/2} + a_{n+1} * dt/2
    """
    tid = wp.tid()
    
    inv_m = inv_mass[tid]
    
    # Acceleration from new forces, gravity, and external forces
    acc = (f[tid] + external_forces[tid]) * inv_m + gravity
    
    # Half kick
    v_new[tid] = v[tid] + acc * (dt / 2.0)


@wp.kernel
def apply_boundary_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    boxsize: float,
):
    """
    Apply reflective boundary conditions.
    
    If a particle exits the box, reflect it back and reverse velocity.
    """
    tid = wp.tid()
    
    pos = x[tid]
    vel = v[tid]
    
    # X dimension
    if pos[0] < 0.0:
        pos = wp.vec2(-pos[0], pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    elif pos[0] > boxsize:
        pos = wp.vec2(boxsize - (pos[0] - boxsize), pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    
    # Y dimension
    if pos[1] < 0.0:
        pos = wp.vec2(pos[0], -pos[1])
        vel = wp.vec2(vel[0], -vel[1])
    elif pos[1] > boxsize:
        pos = wp.vec2(pos[0], boxsize - (pos[1] - boxsize))
        vel = wp.vec2(vel[0], -vel[1])
    
    x[tid] = pos
    v[tid] = vel


@wp.kernel
def apply_boundary_with_friction_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    boxsize: float,
    ground_y: float,         # Ground level (usually 0)
    direction: wp.vec2,      # Locomotion direction (normalized 2D vector)
):
    """
    Apply boundary conditions with DIRECTION-AWARE ratchet friction.
    
    Ratchet mechanism for any direction:
        - Forward (along direction): NO friction - slides freely
        - Backward (against direction): COMPLETE STOP - ratchet
    
    This enables locomotion in ANY 2D direction with symmetric radial forces.
    
    Args:
        ground_y: Y-coordinate of ground (usually 0)
        direction: Normalized 2D locomotion direction vector
    """
    tid = wp.tid()
    
    pos = x[tid]
    vel = v[tid]
    
    # X dimension (walls) - reflective, no friction
    if pos[0] < 0.0:
        pos = wp.vec2(-pos[0], pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    elif pos[0] > boxsize:
        pos = wp.vec2(boxsize - (pos[0] - boxsize), pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    
    # Y dimension - GROUND with DIRECTION-AWARE ratchet friction
    if pos[1] < ground_y:
        # Reflect position
        pos = wp.vec2(pos[0], ground_y - (pos[1] - ground_y))
        
        # Project velocity onto locomotion direction: v_forward = dot(vel, direction)
        v_forward = wp.dot(vel, direction)
        
        if v_forward > 0.0:
            # Moving forward (along direction): NO friction - slides freely
            vx_new = vel[0]
            vy_new = -vel[1]  # Reflect vertical
        else:
            # Moving backward (against direction): STOP motion in direction
            # Remove the backward component: v_new = v - v_backward * direction
            vx_new = vel[0] - v_forward * direction[0]
            vy_new = -vel[1] - v_forward * direction[1]
        
        vel = wp.vec2(vx_new, vy_new)
    
    # Y dimension - ceiling (reflective, no friction)
    elif pos[1] > boxsize:
        pos = wp.vec2(pos[0], boxsize - (pos[1] - boxsize))
        vel = wp.vec2(vel[0], -vel[1])
    
    x[tid] = pos
    v[tid] = vel




# ============================================================================
# High-level wrapper functions (following Newton pattern)
# ============================================================================

def eval_spring_forces_2d(model, state, particle_f: wp.array):
    """
    Evaluate spring forces and compute normalized strains (wrapper function).
    
    Computes both forces and strain metrics in a single kernel launch for efficiency.
    
    Args:
        model: The 2D Model containing spring properties
        state: The current State containing particle positions/velocities
        particle_f: Force accumulation array
    """
    if model.spring_count > 0:
        # Get current strain scale for normalization
        strain_scale = model.spring_strain_scale.numpy()[0]
        
        wp.launch(
            kernel=eval_spring_2d,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                particle_f,
                model.spring_strains,
                model.spring_strains_normalized,
                strain_scale,
            ],
            device=model.device,
        )

