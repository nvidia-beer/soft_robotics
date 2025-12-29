# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Fully Implicit 2D FEM kernels with tangent stiffness matrix computation
# This extends kernels_fem_2d.py by adding FEM stiffness to the system matrix

import warp as wp


# ============================================================================
# Neo-Hookean Tangent Stiffness Computation
# ============================================================================

@wp.func
def compute_neo_hookean_tangent_2d(
    F: wp.mat22,
    Dm_inv: wp.mat22,
    k_mu: float,
    k_lambda: float,
) -> wp.mat22:
    """
    Compute the tangent stiffness contribution for 2D Neo-Hookean material.
    
    Returns a simplified scalar stiffness coefficient that captures the
    dominant behavior of the tangent matrix for use in the system matrix.
    
    For full implicit FEM, we need ∂f/∂x = ∂(P · Dm^T)/∂x
    where P is the first Piola-Kirchhoff stress.
    
    The Neo-Hookean model gives:
        P = μ * F * (I_C - 2) / I_C  (deviatoric)
          + λ * (J - 1) * ∂J/∂F      (volumetric)
    
    The tangent stiffness K = ∂P/∂F is a 4th-order tensor in 2D.
    We approximate it with its dominant eigenvalue for the system matrix.
    """
    # Compute invariants
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)
    J = wp.determinant(F)
    
    # Avoid division by zero
    Ic_safe = wp.max(Ic, 1e-6)
    
    # Deviatoric tangent stiffness (simplified)
    # ∂P_dev/∂F ≈ μ * (1 - 2*(I_C-2)/I_C^2) ≈ μ for small deformations
    K_dev = k_mu * (1.0 + 2.0 / Ic_safe)
    
    # Volumetric tangent stiffness
    # ∂P_vol/∂F = λ * (∂J/∂F ⊗ ∂J/∂F + (J-1) * ∂²J/∂F²)
    # For J ≈ 1: K_vol ≈ λ
    K_vol = k_lambda * (1.0 + wp.abs(J - 1.0))
    
    # Combined effective stiffness
    K_eff = K_dev + K_vol
    
    return wp.mat22(K_eff, 0.0, 0.0, K_eff)


@wp.kernel
def compute_fem_stiffness_blocks_2d(
    x: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=int),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    # Output: triplet format for BSR assembly
    fem_rows: wp.array(dtype=wp.int32),
    fem_cols: wp.array(dtype=wp.int32),
    fem_values: wp.array(dtype=wp.mat22f),
    dt: float,
):
    """
    Compute FEM tangent stiffness matrix blocks for implicit integration.
    
    Each triangle contributes 9 blocks (3x3 vertex pairs):
        K_ij = -h² * ∂f_i/∂x_j
    
    where f_i is the force on vertex i from FEM stress.
    
    The system matrix becomes: A = M - h*D - h²*(K_spring + K_fem)
    
    Block layout per triangle (9 blocks):
        (i,i), (i,j), (i,k)
        (j,i), (j,j), (j,k)
        (k,i), (k,j), (k,k)
    """
    tid = wp.tid()
    
    # Get triangle vertex indices
    i = tri_indices[tid * 3 + 0]
    j = tri_indices[tid * 3 + 1]
    k = tri_indices[tid * 3 + 2]
    
    # Material properties
    mat = tri_materials[tid]
    k_mu = mat[0]
    k_lambda = mat[1]
    # k_damp = mat[2]  # Damping handled separately
    
    # Get current positions
    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    
    # Edge vectors
    x10 = x1 - x0
    x20 = x2 - x0
    
    # Current configuration matrix
    Ds = wp.mat22(
        x10[0], x20[0],
        x10[1], x20[1]
    )
    
    # Rest configuration inverse
    Dm_inv = tri_poses[tid]
    
    # Compute rest area
    det_Dm = wp.determinant(Dm_inv)
    rest_area = 0.5 / wp.abs(det_Dm)
    
    # Scale material by area
    k_mu_scaled = k_mu * rest_area
    k_lambda_scaled = k_lambda * rest_area
    
    # Deformation gradient
    F = Ds * Dm_inv
    
    # ========================================================================
    # Compute Tangent Stiffness Matrix K = ∂f/∂x
    # ========================================================================
    # For Neo-Hookean, the tangent stiffness depends on current deformation.
    # We compute element stiffness matrix Ke and distribute to global K.
    
    # Compute effective stiffness coefficient from Neo-Hookean model
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)
    J = wp.determinant(F)
    
    # Avoid singularities
    Ic_safe = wp.max(Ic, 0.1)
    
    # Deviatoric stiffness: linearization of P = μ * F * (Ic-2)/Ic
    # dP/dF ≈ μ * ((Ic-2)/Ic * I + F ⊗ d((Ic-2)/Ic)/dF)
    # Simplified: K_dev ≈ μ * (1 + 4/Ic) for typical deformations
    K_dev = k_mu_scaled * (1.0 + 4.0 / Ic_safe)
    
    # Volumetric stiffness: linearization of P_vol = λ*(J-1)*cofactor(F)
    # dP_vol/dF ≈ λ * (cofactor ⊗ cofactor/J + (J-1)*d(cofactor)/dF)
    # Simplified: K_vol ≈ λ * (1 + |J-1|)
    K_vol = k_lambda_scaled * (1.0 + wp.abs(J - 1.0) * 2.0)
    
    # Total effective stiffness coefficient
    K_eff = K_dev + K_vol
    
    # Time step scaling for system matrix: h²*K
    dt2 = dt * dt
    K_scaled = dt2 * K_eff
    
    # ========================================================================
    # Build Element Stiffness Matrix
    # ========================================================================
    # For a triangle with vertices (0, 1, 2), the element stiffness matrix
    # relates nodal displacements to nodal forces:
    #   f = -K * u
    #
    # The shape function gradients determine how deformation distributes:
    # B = [∂N/∂x] where N are shape functions
    #
    # For linear triangles with constant strain:
    # B_i = Dm_inv^T * e_i (where e_i are basis vectors)
    
    # Shape function gradient coefficients (from Dm_inv)
    # These determine how each vertex contributes to strain
    b1 = wp.vec2(Dm_inv[0, 0], Dm_inv[1, 0])  # Gradient for vertex 1
    b2 = wp.vec2(Dm_inv[0, 1], Dm_inv[1, 1])  # Gradient for vertex 2
    b0 = -(b1 + b2)  # Gradient for vertex 0 (partition of unity)
    
    # Element stiffness blocks: K_ab = K_eff * (b_a ⊗ b_b)
    # This gives the coupling between vertices a and b
    
    # Compute all 9 blocks using outer products of shape function gradients
    # Block (a,b) = K_eff * b_a ⊗ b_b
    
    # Diagonal blocks (self-coupling)
    K_00 = wp.mat22f(
        K_scaled * b0[0] * b0[0], K_scaled * b0[0] * b0[1],
        K_scaled * b0[1] * b0[0], K_scaled * b0[1] * b0[1]
    )
    K_11 = wp.mat22f(
        K_scaled * b1[0] * b1[0], K_scaled * b1[0] * b1[1],
        K_scaled * b1[1] * b1[0], K_scaled * b1[1] * b1[1]
    )
    K_22 = wp.mat22f(
        K_scaled * b2[0] * b2[0], K_scaled * b2[0] * b2[1],
        K_scaled * b2[1] * b2[0], K_scaled * b2[1] * b2[1]
    )
    
    # Off-diagonal blocks (cross-coupling)
    K_01 = wp.mat22f(
        K_scaled * b0[0] * b1[0], K_scaled * b0[0] * b1[1],
        K_scaled * b0[1] * b1[0], K_scaled * b0[1] * b1[1]
    )
    K_02 = wp.mat22f(
        K_scaled * b0[0] * b2[0], K_scaled * b0[0] * b2[1],
        K_scaled * b0[1] * b2[0], K_scaled * b0[1] * b2[1]
    )
    K_12 = wp.mat22f(
        K_scaled * b1[0] * b2[0], K_scaled * b1[0] * b2[1],
        K_scaled * b1[1] * b2[0], K_scaled * b1[1] * b2[1]
    )
    
    # Symmetric: K_10 = K_01^T, etc.
    K_10 = wp.transpose(K_01)
    K_20 = wp.transpose(K_02)
    K_21 = wp.transpose(K_12)
    
    # ========================================================================
    # Store blocks in triplet format (9 blocks per triangle)
    # ========================================================================
    block_idx = tid * 9
    
    # Row 0 (vertex i)
    fem_rows[block_idx + 0] = i
    fem_cols[block_idx + 0] = i
    fem_values[block_idx + 0] = K_00
    
    fem_rows[block_idx + 1] = i
    fem_cols[block_idx + 1] = j
    fem_values[block_idx + 1] = K_01
    
    fem_rows[block_idx + 2] = i
    fem_cols[block_idx + 2] = k
    fem_values[block_idx + 2] = K_02
    
    # Row 1 (vertex j)
    fem_rows[block_idx + 3] = j
    fem_cols[block_idx + 3] = i
    fem_values[block_idx + 3] = K_10
    
    fem_rows[block_idx + 4] = j
    fem_cols[block_idx + 4] = j
    fem_values[block_idx + 4] = K_11
    
    fem_rows[block_idx + 5] = j
    fem_cols[block_idx + 5] = k
    fem_values[block_idx + 5] = K_12
    
    # Row 2 (vertex k)
    fem_rows[block_idx + 6] = k
    fem_cols[block_idx + 6] = i
    fem_values[block_idx + 6] = K_20
    
    fem_rows[block_idx + 7] = k
    fem_cols[block_idx + 7] = j
    fem_values[block_idx + 7] = K_21
    
    fem_rows[block_idx + 8] = k
    fem_cols[block_idx + 8] = k
    fem_values[block_idx + 8] = K_22


@wp.kernel
def build_combined_system_matrix_2d(
    # Spring contributions
    spring_rows: wp.array(dtype=wp.int32),
    spring_cols: wp.array(dtype=wp.int32),
    spring_values: wp.array(dtype=wp.mat22f),
    spring_indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # Parameters
    dt: float,
    mass: float,
    Minv: float,
):
    """
    Build spring contribution to system matrix.
    
    Same as build_system_matrix_sparse_2d but separate for clarity.
    FEM blocks are computed separately and combined during BSR assembly.
    """
    tid = wp.tid()
    
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    
    k = spring_stiffness[tid]
    d = spring_damping[tid]
    
    dt2 = dt * dt
    dt2_k = dt2 * k
    dt_d_Minv = dt * d * Minv
    
    # Diagonal: M + h*D*M^-1 - h²*K
    diag_coeff = mass + dt_d_Minv - dt2_k
    block_diag = wp.mat22f(
        diag_coeff, 0.0,
        0.0, diag_coeff
    )
    
    # Off-diagonal: h²*K
    offdiag_coeff = dt2_k
    block_offdiag = wp.mat22f(
        offdiag_coeff, 0.0,
        0.0, offdiag_coeff
    )
    
    block_idx = tid * 4
    
    spring_rows[block_idx] = i
    spring_cols[block_idx] = i
    spring_values[block_idx] = block_diag
    
    spring_rows[block_idx + 1] = j
    spring_cols[block_idx + 1] = j
    spring_values[block_idx + 1] = block_diag
    
    spring_rows[block_idx + 2] = i
    spring_cols[block_idx + 2] = j
    spring_values[block_idx + 2] = block_offdiag
    
    spring_rows[block_idx + 3] = j
    spring_cols[block_idx + 3] = i
    spring_values[block_idx + 3] = block_offdiag


@wp.kernel
def eval_triangles_fem_implicit_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    indices: wp.array(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    materials: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec2),
    tri_strains: wp.array(dtype=float),
    tri_strains_normalized: wp.array(dtype=float),
    strain_scale: float,
):
    """
    Evaluate FEM forces for fully implicit solver.
    
    Same as eval_triangles_fem_2d but forces are scaled for implicit RHS.
    The stiffness contribution is handled separately in the system matrix.
    """
    tid = wp.tid()
    
    # Get triangle vertex indices
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]
    
    # Material properties
    mat = materials[tid]
    k_mu = mat[0]
    k_lambda = mat[1]
    k_damp = mat[2]
    
    # Get positions and velocities
    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    
    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    
    # Edge vectors
    x10 = x1 - x0
    x20 = x2 - x0
    v10 = v1 - v0
    v20 = v2 - v0
    
    # Configuration matrices
    Ds = wp.mat22(x10[0], x20[0], x10[1], x20[1])
    Dm = pose[tid]
    
    det_Dm = wp.determinant(Dm)
    rest_area = 0.5 / wp.abs(det_Dm)
    
    # Scale materials
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area
    
    # Deformation gradient
    F = Ds * Dm
    dFdt = wp.mat22(v10[0], v20[0], v10[1], v20[1]) * Dm
    
    # Deviatoric
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)
    
    # First Piola-Kirchhoff stress (deviatoric + damping)
    P = F * k_mu * (Ic - 2.0) / Ic + dFdt * k_damp
    H = P * wp.transpose(Dm)
    
    f1 = wp.vec2(H[0, 0], H[1, 0])
    f2 = wp.vec2(H[0, 1], H[1, 1])
    
    # Volumetric
    J = wp.determinant(F)
    dJdx1 = wp.vec2(x20[1], -x20[0]) * det_Dm
    dJdx2 = wp.vec2(-x10[1], x10[0]) * det_Dm
    
    f_volume = (J - 1.0) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2)) * k_damp
    f_total = f_volume + f_damp
    
    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f0 = -(f1 + f2)
    
    # Strain computation - SIGNED (centered at 0 when at rest)
    # J > 1 → expansion → positive strain
    # J < 1 → compression → negative strain
    # J = 1 → at rest → zero strain
    deviatoric_strain = (Ic - 2.0) / 2.0  # Signed
    volumetric_strain = J - 1.0           # Signed
    
    k_mu_orig = k_mu / rest_area
    k_lambda_orig = k_lambda / rest_area
    total_stiffness = k_mu_orig + k_lambda_orig
    
    w_deviatoric = k_mu_orig / total_stiffness
    w_volumetric = k_lambda_orig / total_stiffness
    
    total_strain = w_deviatoric * deviatoric_strain + w_volumetric * volumetric_strain
    tri_strains[tid] = total_strain
    
    safe_scale = wp.max(strain_scale, 1e-8)
    
    # Dead-zone: ignore strains with magnitude below noise threshold (1e-5)
    noise_threshold = 1e-5
    effective_strain = wp.where(wp.abs(total_strain) < noise_threshold, 0.0, total_strain)
    
    normalized = wp.clamp(effective_strain / safe_scale, -1.0, 1.0)
    tri_strains_normalized[tid] = normalized
    
    # Apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


# ============================================================================
# Wrapper Functions
# ============================================================================

def compute_fem_stiffness_2d(model, state, dt: float, fem_rows, fem_cols, fem_values):
    """
    Compute FEM tangent stiffness blocks for implicit system matrix.
    
    Args:
        model: Model with triangle data
        state: Current state (positions)
        dt: Time step
        fem_rows, fem_cols, fem_values: Output triplet arrays
    """
    if hasattr(model, 'tri_count') and model.tri_count > 0:
        wp.launch(
            kernel=compute_fem_stiffness_blocks_2d,
            dim=model.tri_count,
            inputs=[
                state.particle_q,
                model.tri_indices,
                model.tri_poses,
                model.tri_materials,
                fem_rows,
                fem_cols,
                fem_values,
                dt,
            ],
            device=model.device,
        )


def eval_triangle_fem_forces_implicit_2d(model, state, particle_f):
    """Evaluate FEM forces for implicit solver."""
    if hasattr(model, 'tri_count') and model.tri_count > 0:
        fem_strain_scale = model.fem_strain_scale.numpy()[0]
        
        wp.launch(
            kernel=eval_triangles_fem_implicit_2d,
            dim=model.tri_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_poses,
                model.tri_materials,
                particle_f,
                model.tri_strains,
                model.tri_strains_normalized,
                fem_strain_scale,
            ],
            device=model.device,
        )














