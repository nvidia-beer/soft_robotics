# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# 2D VBD (Vertex Block Descent) Kernels
# Implements gradient and Hessian computation for implicit optimization
#
# Based on: Chen et al. 2024, "Vertex Block Descent" (SIGGRAPH)
# Theory: Minimize total potential E = E_inertia + E_elastic per vertex
#
# Key equations (2D adaptation):
# ------------------------------
# E_inertia = (1/2h²) * m * ||x - y||² where y = x_prev + h*v_prev
# E_elastic = Neo-Hookean strain energy (stable variant)
#
# ∇E_inertia = (m/h²) * (x - y)
# H_inertia = (m/h²) * I₂
#
# For elastic: ∇E and H computed from deformation gradient F = Ds * Dm⁻¹

import warp as wp

EPS = 1e-12


# ============================================================================
# Inertia Gradient and Hessian (2D)
# ============================================================================

@wp.func
def inertial_gradient_hessian_2d(
    vertex_idx: wp.int32,
    tri_idx: wp.int32,
    positions: wp.array(dtype=wp.vec2),
    old_positions: wp.array(dtype=wp.vec2),
    old_velocities: wp.array(dtype=wp.vec2),
    tri_masses: wp.array(dtype=wp.float32),
    gravity: wp.vec2,
    dt: wp.float32,
):
    """
    Compute inertia gradient and Hessian for a vertex from one triangle.
    
    Inertia potential (implicit Euler):
        E_inertia = (1/2h²) * m * ||x - y||²
        where y = x_prev + h*v_prev + h²*g (predicted position)
    
    Gradient:
        ∇E = (m/h²) * (x - y) - m*g
    
    Hessian:
        H = (m/h²) * I₂
    
    Note: Mass is distributed equally among triangle vertices (m_tri / 3).
    """
    x = positions[vertex_idx]
    x_prev = old_positions[vertex_idx]
    v_prev = old_velocities[vertex_idx]
    
    # Mass per vertex from this triangle (1/3 of triangle mass)
    m = tri_masses[tri_idx] / 3.0
    
    # Initialize gradient and Hessian
    gradient = wp.vec2(0.0, 0.0)
    hessian = wp.mat22(0.0, 0.0, 0.0, 0.0)
    
    # Predicted position (explicit Euler step)
    y = x_prev + dt * v_prev
    
    # Inertia gradient: (m/h²) * (x - y)
    h2_inv = 1.0 / (dt * dt)
    gradient = m * h2_inv * (x - y)
    
    # Inertia Hessian: (m/h²) * I₂
    hess_coeff = m * h2_inv
    hessian = wp.mat22(hess_coeff, 0.0, 0.0, hess_coeff)
    
    # Gravity (external force -> negative gradient)
    gradient = gradient - m * gravity
    # No Hessian contribution from gravity
    
    return gradient, hessian


# ============================================================================
# Elastic Gradient and Hessian - Stable Neo-Hookean 2D
# ============================================================================

@wp.func
def elastic_gradient_hessian_2d(
    vertex_idx: wp.int32,
    tri_idx: wp.int32,
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    tri_areas: wp.array(dtype=wp.float32),
):
    """
    Compute elastic gradient and Hessian for a vertex from one triangle.
    
    Uses Stable Neo-Hookean model (Smith et al. 2018) adapted for 2D:
    
    Energy density:
        Ψ = (μ/2) * (I_C - 2 - 2*ln(J)) + (λ/2) * (J - 1)²
    
    where:
        F = Ds * Dm⁻¹  (deformation gradient)
        I_C = tr(F^T * F) = ||F||²_F (first invariant, = 2 at rest)
        J = det(F) (area ratio, = 1 at rest)
    
    First Piola-Kirchhoff stress (2D Stable Neo-Hookean):
        P = μ * F * (1 - 1/I_C) + λ * (J - α) * J * F⁻ᵀ
        where α = 1 + (3/4)*μ/λ
    
    Returns gradient and Hessian with respect to vertex position.
    """
    # Get triangle vertex indices
    i = tri_indices[tri_idx * 3 + 0]
    j = tri_indices[tri_idx * 3 + 1]
    k = tri_indices[tri_idx * 3 + 2]
    
    # Get positions
    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    
    # Material properties
    mat = tri_materials[tri_idx]
    k_mu = mat[0]
    k_lambda = mat[1]
    k_damp = mat[2]  # Unused in elastic energy (handled separately)
    
    # Rest configuration inverse
    Dm_inv = tri_poses[tri_idx]
    
    # Rest area
    rest_area = tri_areas[tri_idx]
    
    # Build current configuration matrix Ds = [x10, x20]
    x10 = x1 - x0
    x20 = x2 - x0
    Ds = wp.mat22(
        x10[0], x20[0],
        x10[1], x20[1]
    )
    
    # Deformation gradient F = Ds * Dm⁻¹
    F = Ds * Dm_inv
    
    # ========================================================================
    # Compute invariants
    # ========================================================================
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)  # I_C = ||F||² (= 2 at rest)
    
    J = wp.determinant(F)  # J = det(F) (= 1 at rest)
    
    # Avoid degenerate configurations
    Ic_safe = wp.max(Ic, 0.01)
    J_safe = wp.max(wp.abs(J), 1e-8)
    
    # F inverse transpose for volumetric terms
    # For 2D: F⁻ᵀ = (1/J) * [[F22, -F21], [-F12, F11]]
    F_inv_T = wp.mat22(
        F[1, 1] / J_safe, -F[1, 0] / J_safe,
        -F[0, 1] / J_safe, F[0, 0] / J_safe
    )
    
    # ========================================================================
    # 2D Neo-Hookean Stress (same as implicit FEM, zero at rest)
    # ========================================================================
    # Deviatoric: P_dev = μ * F * (I_C - 2) / I_C
    #   At rest (F=I, I_C=2): P_dev = μ * I * 0 = 0 ✓
    # Volumetric: P_vol = λ * (J - 1) * F^{-T}
    #   At rest (F=I, J=1): P_vol = λ * 0 * I = 0 ✓
    
    P_dev = k_mu * (Ic - 2.0) / Ic_safe * F
    P_vol = k_lambda * (J - 1.0) * F_inv_T
    P = P_dev + P_vol
    
    # ========================================================================
    # Compute shape function gradients (derivatives of F w.r.t. vertex positions)
    # ========================================================================
    # For linear triangles, ∂F/∂x_i depends on which vertex
    # ∂F/∂x = ∂(Ds * Dm⁻¹)/∂x = ∂Ds/∂x * Dm⁻¹
    
    # Build ∂Ds/∂x for each vertex
    # Ds = [[x10.x, x20.x], [x10.y, x20.y]] where x10 = x1 - x0, x20 = x2 - x0
    #
    # For vertex 0 (x0): ∂Ds/∂x0 = -I for both columns
    # For vertex 1 (x1): ∂Ds/∂x1 affects only first column
    # For vertex 2 (x2): ∂Ds/∂x2 affects only second column
    
    # Determine which vertex we're computing gradient for
    is_v0 = wp.float32(vertex_idx == i)
    is_v1 = wp.float32(vertex_idx == j)
    is_v2 = wp.float32(vertex_idx == k)
    
    # Shape function gradients in reference space (B matrix)
    # These come from: H = P * Dm_inv^T, and force_i = H[:, i]
    # So gradient_i = P @ (row i of Dm_inv, as column vector)
    # b1 = first row of Dm_inv (for vertex 1)
    # b2 = second row of Dm_inv (for vertex 2)
    # b0 = -(b1 + b2) (partition of unity)
    b1 = wp.vec2(Dm_inv[0, 0], Dm_inv[0, 1])  # First ROW of Dm_inv
    b2 = wp.vec2(Dm_inv[1, 0], Dm_inv[1, 1])  # Second ROW of Dm_inv
    b0 = -(b1 + b2)
    
    # Select appropriate shape function gradient
    b = is_v0 * b0 + is_v1 * b1 + is_v2 * b2
    
    # ========================================================================
    # Gradient: ∇E = V * P * b  (matrix-vector product)
    # ========================================================================
    # Explicit matrix-vector multiplication for clarity
    grad_x = rest_area * (P[0, 0] * b[0] + P[0, 1] * b[1])
    grad_y = rest_area * (P[1, 0] * b[0] + P[1, 1] * b[1])
    gradient = wp.vec2(grad_x, grad_y)
    
    # ========================================================================
    # Hessian: State-dependent diagonal approximation
    # ========================================================================
    # The Hessian should scale with actual deformation, not just material stiffness
    # At rest (F=I, J=1, Ic=2), the Hessian should be small
    
    b_norm_sq = b[0] * b[0] + b[1] * b[1]
    
    # Strain-dependent stiffness: increases with deformation
    # At rest: Ic=2, J=1 -> strain_factor ≈ 0.1 (small but nonzero for stability)
    # Deformed: strain_factor increases
    strain_measure = wp.abs(Ic - 2.0) + wp.abs(J - 1.0)
    strain_factor = wp.min(strain_measure + 0.001, 2.0)  # Clamp for stability
    
    # Material stiffness scaled by strain (use original k_mu, k_lambda)
    K_material = rest_area * (k_mu + k_lambda) * b_norm_sq * strain_factor
    
    # Use a positive definite diagonal Hessian
    hessian = wp.mat22(
        K_material, 0.0,
        0.0, K_material
    )
    
    return gradient, hessian


# ============================================================================
# Damping Gradient and Hessian
# ============================================================================

@wp.func
def damping_gradient_hessian_2d(
    vertex_idx: wp.int32,
    tri_idx: wp.int32,
    positions: wp.array(dtype=wp.vec2),
    old_positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    tri_areas: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """
    Compute damping gradient and Hessian.
    
    Damping force is proportional to elastic Hessian times velocity:
        f_damp = -D * K * (x - x_prev) / dt
    
    This provides energy dissipation while maintaining stability.
    """
    # Get triangle vertex indices
    i = tri_indices[tri_idx * 3 + 0]
    j = tri_indices[tri_idx * 3 + 1]
    k = tri_indices[tri_idx * 3 + 2]
    
    # Material properties (damping coefficient)
    mat = tri_materials[tri_idx]
    k_damp = mat[2]
    
    # Get the elastic Hessian estimate (simplified version)
    k_mu = mat[0]
    k_lambda = mat[1]
    rest_area = tri_areas[tri_idx]
    
    # Estimate stiffness for damping
    K_eff = rest_area * (k_mu + k_lambda)
    
    # Velocity estimate
    x = positions[vertex_idx]
    x_prev = old_positions[vertex_idx]
    v_est = (x - x_prev) / dt
    
    # Damping gradient: D * K * v
    damping_coeff = k_damp * K_eff
    gradient = damping_coeff * v_est
    
    # Damping Hessian: D * K / dt
    hess_coeff = damping_coeff / dt
    hessian = wp.mat22(hess_coeff, 0.0, 0.0, hess_coeff)
    
    return gradient, hessian


# ============================================================================
# Main VBD Solve Kernel
# ============================================================================

@wp.kernel
def vbd_solve_vertex_2d(
    positions: wp.array(dtype=wp.vec2),
    old_positions: wp.array(dtype=wp.vec2),
    old_velocities: wp.array(dtype=wp.vec2),
    
    tri_indices: wp.array(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    tri_masses: wp.array(dtype=wp.float32),
    tri_areas: wp.array(dtype=wp.float32),
    
    gravity: wp.vec2,
    dt: wp.float32,
    damping_coefficient: wp.float32,
    
    adj_v2t: wp.array2d(dtype=wp.int32),
    color_group: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.int32),
    
    new_positions: wp.array(dtype=wp.vec2),
    grads: wp.array(dtype=wp.vec2),
    dxs: wp.array(dtype=wp.vec2),
):
    """
    Main VBD kernel: solve local linear system for each vertex in color group.
    
    Algorithm (per vertex):
    1. Accumulate gradient and Hessian from all incident triangles
    2. Solve 2x2 system: H * dx = -g
    3. Update position: x_new = x + dx
    
    This kernel is launched once per color group. Vertices in the same
    color group are non-adjacent and can be updated in parallel.
    
    Theory (VBD Paper):
    ------------------
    VBD minimizes the implicit Euler objective by coordinate descent:
        E(x) = Σ_v [E_inertia(x_v) + E_elastic(x_v)]
    
    Per-vertex optimization uses Newton's method with block-diagonal Hessian:
        dx = -H_v⁻¹ * ∇E_v
    
    Graph coloring ensures parallel updates don't interfere.
    """
    tid = wp.tid()
    
    # Get vertex index from color group
    idx_v = color_group[tid]
    if idx_v == -1:
        return
    
    # Skip inactive vertices (boundary conditions)
    if active_mask[idx_v] == 0:
        return
    
    # Initialize accumulators
    grad = wp.vec2(0.0, 0.0)
    hess = wp.mat22(0.0, 0.0, 0.0, 0.0)
    
    # Loop over incident triangles
    num_incident = adj_v2t.shape[1]
    for j in range(num_incident):
        tri_idx = adj_v2t[idx_v, j]
        if tri_idx == -1:
            continue
        
        # Inertia contribution
        grad_inertia, hess_inertia = inertial_gradient_hessian_2d(
            idx_v, tri_idx,
            positions, old_positions, old_velocities,
            tri_masses, gravity, dt
        )
        
        # Elastic contribution
        grad_elastic, hess_elastic = elastic_gradient_hessian_2d(
            idx_v, tri_idx,
            positions, tri_indices, tri_poses, tri_materials, tri_areas
        )
        
        # Damping contribution
        grad_damp, hess_damp = damping_gradient_hessian_2d(
            idx_v, tri_idx,
            positions, old_positions,
            tri_indices, tri_poses, tri_materials, tri_areas, dt
        )
        
        # Scale damping
        grad_damp = damping_coefficient * grad_damp
        hess_damp = damping_coefficient * hess_damp
        
        # Accumulate
        grad = grad + grad_inertia + grad_elastic + grad_damp
        hess = hess + hess_inertia + hess_elastic + hess_damp
    
    # Solve local 2x2 linear system: H * dx = -g
    # Add regularization for numerical stability (scaled by typical Hessian magnitude)
    hess_diag_avg = 0.5 * (wp.abs(hess[0, 0]) + wp.abs(hess[1, 1]))
    reg = wp.max(hess_diag_avg * 0.01, 1e-4)  # 1% of diagonal or minimum 1e-4
    
    hess_reg = wp.mat22(
        hess[0, 0] + reg, hess[0, 1],
        hess[1, 0], hess[1, 1] + reg
    )
    
    # 2x2 inverse (explicit formula)
    det = hess_reg[0, 0] * hess_reg[1, 1] - hess_reg[0, 1] * hess_reg[1, 0]
    det_safe = wp.max(wp.abs(det), 1e-6)
    
    inv_det = 1.0 / det_safe
    hess_inv = wp.mat22(
        hess_reg[1, 1] * inv_det, -hess_reg[0, 1] * inv_det,
        -hess_reg[1, 0] * inv_det, hess_reg[0, 0] * inv_det
    )
    
    # Compute position update
    dx = -(hess_inv * grad)
    
    # Store results
    grads[idx_v] = grad
    dxs[idx_v] = dx
    new_positions[idx_v] = positions[idx_v] + dx


# ============================================================================
# Position Initialization Kernel
# ============================================================================

@wp.kernel
def vbd_position_init_2d(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    gravity: wp.vec2,
    dt: wp.float32,
    active_mask: wp.array(dtype=wp.int32),
    new_positions: wp.array(dtype=wp.vec2),
):
    """
    Initialize positions with explicit Euler prediction.
    
    x_init = x + v*dt + g*dt²
    
    This provides a good initial guess for VBD iterations.
    """
    i = wp.tid()
    
    if active_mask[i] == 0:
        # Inactive vertex: keep position unchanged
        new_positions[i] = positions[i]
        return
    
    # Explicit Euler prediction
    new_positions[i] = positions[i] + velocities[i] * dt + gravity * dt * dt


# ============================================================================
# Triangle Mass Computation
# ============================================================================

@wp.kernel
def compute_triangle_masses_2d(
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.int32),
    densities: wp.array(dtype=wp.float32),
    tri_masses: wp.array(dtype=wp.float32),
    tri_areas: wp.array(dtype=wp.float32),
):
    """
    Compute triangle masses from positions and densities.
    
    Mass = density * area
    Area = 0.5 * |det([x10, x20])|
    """
    tid = wp.tid()
    
    # Get triangle vertices
    i = tri_indices[tid * 3 + 0]
    j = tri_indices[tid * 3 + 1]
    k = tri_indices[tid * 3 + 2]
    
    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    
    # Compute area
    x10 = x1 - x0
    x20 = x2 - x0
    cross = x10[0] * x20[1] - x10[1] * x20[0]
    area = 0.5 * wp.abs(cross)
    
    # Store area and mass
    tri_areas[tid] = area
    tri_masses[tid] = densities[tid] * area


@wp.kernel
def compute_masses_from_areas_2d(
    tri_areas: wp.array(dtype=wp.float32),
    densities: wp.array(dtype=wp.float32),
    tri_masses: wp.array(dtype=wp.float32),
):
    """
    Compute triangle masses from pre-computed areas.
    
    Mass = density * area
    """
    tid = wp.tid()
    tri_masses[tid] = densities[tid] * tri_areas[tid]


# ============================================================================
# Strain Computation Kernel (for visualization)
# ============================================================================

@wp.kernel
def compute_triangle_strains_2d(
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.int32),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    tri_strains: wp.array(dtype=wp.float32),
    tri_strains_normalized: wp.array(dtype=wp.float32),
    strain_scale: wp.float32,
):
    """
    Compute strain metrics for visualization.
    
    Uses combined deviatoric + volumetric strain weighted by material properties.
    """
    tid = wp.tid()
    
    # Get triangle vertices
    i = tri_indices[tid * 3 + 0]
    j = tri_indices[tid * 3 + 1]
    k = tri_indices[tid * 3 + 2]
    
    x0 = positions[i]
    x1 = positions[j]
    x2 = positions[k]
    
    # Build deformation gradient
    x10 = x1 - x0
    x20 = x2 - x0
    Ds = wp.mat22(x10[0], x20[0], x10[1], x20[1])
    Dm_inv = tri_poses[tid]
    F = Ds * Dm_inv
    
    # Compute invariants
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)
    J = wp.determinant(F)
    
    # Material properties for weighting
    mat = tri_materials[tid]
    k_mu = mat[0]
    k_lambda = mat[1]
    total_k = k_mu + k_lambda
    
    # Strain components (signed)
    deviatoric_strain = (Ic - 2.0) / 2.0
    volumetric_strain = J - 1.0
    
    # Weighted combination
    w_dev = k_mu / total_k
    w_vol = k_lambda / total_k
    total_strain = w_dev * deviatoric_strain + w_vol * volumetric_strain
    
    # Store raw strain
    tri_strains[tid] = total_strain
    
    # Normalize to [-1, 1]
    safe_scale = wp.max(strain_scale, 1e-8)
    normalized = wp.clamp(total_strain / safe_scale, -1.0, 1.0)
    tri_strains_normalized[tid] = normalized


# ============================================================================
# Spring Strain Computation (for visualization only, VBD doesn't use springs)
# ============================================================================

@wp.kernel
def compute_spring_strains_2d(
    positions: wp.array(dtype=wp.vec2),
    spring_indices: wp.array(dtype=wp.int32),
    spring_rest_lengths: wp.array(dtype=wp.float32),
    spring_strains: wp.array(dtype=wp.float32),
    spring_strains_normalized: wp.array(dtype=wp.float32),
    strain_scale: wp.float32,
):
    """
    Compute spring strains for visualization (no force computation).
    VBD uses FEM for physics, this is just for display.
    """
    tid = wp.tid()
    
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    
    if i == -1 or j == -1:
        return
    
    xi = positions[i]
    xj = positions[j]
    
    xij = xi - xj
    l = wp.length(xij)
    rest = spring_rest_lengths[tid]
    
    if l < 1e-6:
        spring_strains[tid] = 0.0
        spring_strains_normalized[tid] = 0.0
        return
    
    # Raw strain: ε = (L - L₀) / L₀
    raw_strain = (l - rest) / rest
    spring_strains[tid] = raw_strain
    
    # Normalize to [-1, 1]
    safe_scale = wp.max(strain_scale, 1e-8)
    normalized = wp.clamp(raw_strain / safe_scale, -1.0, 1.0)
    spring_strains_normalized[tid] = normalized


# ============================================================================
# Boundary Condition Kernel
# ============================================================================

@wp.kernel
def apply_boundary_vbd_2d(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    boxsize: wp.float32,
):
    """
    Apply reflective boundary conditions.
    """
    tid = wp.tid()
    
    pos = positions[tid]
    vel = velocities[tid]
    
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
    
    positions[tid] = pos
    velocities[tid] = vel
