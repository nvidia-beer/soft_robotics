# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# 2D FEM kernels for implicit solver with triangular elements
# Adapted from soft/kernels.py (tetrahedral FEM -> triangular FEM)

import warp as wp


@wp.kernel
def eval_triangles_fem_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    indices: wp.array(dtype=int),  # Flattened: [i0, j0, k0, i1, j1, k1, ...]
    pose: wp.array(dtype=wp.mat22),  # Rest configuration inverse
    materials: wp.array(dtype=wp.vec3),  # [k_mu, k_lambda, k_damp] per triangle
    f: wp.array(dtype=wp.vec2),
    tri_strains: wp.array(dtype=float),  # Output: raw strain
    tri_strains_normalized: wp.array(dtype=float),  # Output: normalized strain [-1, 1]
    strain_scale: float,  # Adaptive normalization scale
):
    """
    2D Triangular FEM with Neo-Hookean Material Model
    ==================================================
    
    This kernel implements Finite Element Method (FEM) for 2D triangular elements using
    the Neo-Hookean hyperelastic material model with rest stability from Smith et al. 2018.
    
    Physical Interpretation:
    -----------------------
    - Models elastic membranes/fabrics that can undergo large deformations
    - Separates deformation into two parts:
      1. Deviatoric (shape change): stretching, shearing - controlled by k_mu
      2. Volumetric (area change): expansion, compression - controlled by k_lambda
    
    Algorithm Overview:
    ------------------
    1. Compute deformation gradient F = Ds * Dm^-1
       - Ds: current edge vectors [x10, x20]
       - Dm: rest edge vectors inverse (pre-computed)
       - F describes local deformation (stretch, rotation, shear)
    
    2. Compute deviatoric forces (shape preservation):
       - I_C = trace(F^T * F) = sum of squared singular values
       - Measures total stretching/compression
       - Force opposes deviation from rest shape
    
    3. Compute volumetric forces (area preservation):
       - J = det(F) = current_area / rest_area
       - J < 1: compressed, J > 1: expanded, J = 1: equilibrium
       - Force drives J toward rest value 1.0
    
    4. Add damping forces proportional to deformation velocity
    
    References:
    ----------
    - Original 3D tetrahedral implementation: soft/kernels.py:eval_tetrahedra
    - Rest stability: Smith et al. 2018, "Stable Neo-Hookean Flesh Simulation"
    
    Args:
        x: Current particle positions (vec2)
        v: Current particle velocities (vec2)
        indices: Triangle vertex indices [3*tri_count], flattened
        pose: Rest configuration Dm^-1 for each triangle (mat22)
        materials: Material properties [k_mu, k_lambda, k_damp] per triangle
        f: Output forces (vec2)
    
    Material Properties:
        k_mu: Shear modulus (μ) - controls resistance to shape deformation (deviatoric)
        k_lambda: Bulk modulus (λ) - controls resistance to area change (volumetric)
        k_damp: Damping coefficient - controls energy dissipation
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
    
    # Get current positions and velocities
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
    
    # ========================================================================
    # STEP 1: Build Deformation Gradient F = Ds * Dm^-1
    # ========================================================================
    # Current configuration matrix: Ds = [x10, x20] as columns
    # Maps from reference space to current (deformed) space
    Ds = wp.mat22(
        x10[0], x20[0],
        x10[1], x20[1]
    )
    
    # Rest configuration inverse: Dm^-1 (pre-computed and stored)
    # Maps from reference space to rest (undeformed) space
    Dm = pose[tid]
    
    # Compute rest area from stored inverse rest configuration
    # Triangle area = 0.5 * |det(Dm^-1)|
    # Therefore: inv_rest_area = 1 / (0.5 * |det(Dm^-1)|) = 2 * det(Dm)
    # But we store det(Dm) directly for cleaner gradient calculations
    det_Dm = wp.determinant(Dm)
    rest_area = 0.5 / wp.abs(det_Dm)
    
    # NOTE: Removed incorrect "alpha" parameter that was causing spurious forces at rest
    # At rest configuration (F = I), we have J = 1.0 and should have zero volumetric force
    # The previous alpha formula caused artificial expansion/compression forces
    # Now using standard Neo-Hookean model: equilibrium at J = 1.0
    
    # Scale material coefficients by element area
    # Larger triangles contribute more force (total stiffness = density * area)
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area
    
    # ========================================================================
    # STEP 2: Compute Deformation Gradient and Its Time Derivative
    # ========================================================================
    # Deformation gradient: F = Ds * Dm^-1
    # F maps vectors from rest configuration to current configuration
    # Contains stretching, rotation, and shearing information
    F = Ds * Dm
    
    # Velocity gradient: dF/dt for damping forces
    # Rate of change of deformation
    dFdt = wp.mat22(
        v10[0], v20[0],
        v10[1], v20[1]
    ) * Dm
    
    # ========================================================================
    # STEP 3: Deviatoric Energy (Shape Preservation)
    # ========================================================================
    # Neo-Hookean deviatoric strain measure: I_C = trace(F^T * F)
    # I_C = sum of squared singular values of F
    # I_C = 2 at rest (2D: two orthogonal directions, F = Identity)
    # I_C > 2: stretching/shearing
    # I_C < 2: compression (physically possible in 2D)
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2)
    
    # First Piola-Kirchhoff stress tensor (deviatoric + damping)
    # Standard Neo-Hookean deviatoric stress: P = μ(F - F^-T)
    # Simplified form using I_C for numerical stability: P = μ * F * (I_C - 2) / I_C
    # At rest: I_C = 2 → coefficient = 0 → P = 0 (zero stress, as it should be!)
    # Stretched: I_C > 2 → positive stress (resistance to stretching)
    # Note: Division by I_C provides stability (prevents singularity at F=0)
    P = F * k_mu * (Ic - 2.0) / Ic + dFdt * k_damp
    
    # Transform stress to force gradients: H = P * Dm^T
    # H columns give force contributions to vertices 1 and 2
    H = P * wp.transpose(Dm)
    
    # Extract force vectors from H matrix columns
    f1 = wp.vec2(H[0, 0], H[1, 0])
    f2 = wp.vec2(H[0, 1], H[1, 1])
    
    # ========================================================================
    # STEP 4: Volumetric Energy (Area Preservation)
    # ========================================================================
    # Jacobian determinant: J = det(F) = current_area / rest_area
    # J = 1: no area change (equilibrium, rest configuration)
    # J < 1: compressed (area decreased)
    # J > 1: expanded (area increased)
    J = wp.determinant(F)
    
    # Compute gradients dJ/dx_i for each vertex
    # These are perpendicular to opposite edges and point "outward"
    # Derived from chain rule: dJ/dx = d(det(Ds))/dx * det(Dm)
    #
    # For 2D determinant of matrix with columns [x10, x20]:
    #   det = x10.x * x20.y - x10.y * x20.x
    #
    # Partial derivatives:
    #   ∂det/∂x1 = (∂/∂x1)[x10.x * x20.y - x10.y * x20.x]
    #            = (x20.y, -x20.x)  [perpendicular to x20, rotated 90° CCW]
    #   ∂det/∂x2 = (∂/∂x2)[x10.x * x20.y - x10.y * x20.x]
    #            = (-x10.y, x10.x)  [perpendicular to x10, rotated 90° CW]
    #
    # Scale by det(Dm) to get dJ/dx:
    dJdx1 = wp.vec2(x20[1], -x20[0]) * det_Dm
    dJdx2 = wp.vec2(-x10[1], x10[0]) * det_Dm
    
    # Volumetric force magnitude: drives J toward 1.0 (rest configuration)
    # f_mag = (J - 1.0) * k_lambda
    # When J < 1.0: compressed → positive force → expansion
    # When J > 1.0: expanded → negative force → compression
    # At rest (J = 1.0): force = 0 (as it should be!)
    f_volume = (J - 1.0) * k_lambda
    
    # Area damping: dissipates energy from area oscillations
    # Proportional to rate of area change: dJ/dt = Σ (dJ/dx_i · v_i)
    # Note: vertex 0 contribution cancels out due to f0 = -(f1 + f2)
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2)) * k_damp
    
    # Total volumetric force coefficient
    f_total = f_volume + f_damp
    
    # Add volumetric forces to deviatoric forces
    # Force on vertex i = dJ/dx_i * f_total
    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    
    # Force on vertex 0: Newton's third law (total force = 0)
    f0 = -(f1 + f2)
    
    # ========================================================================
    # STEP 4.5: Compute Combined Strain Metric for Monitoring
    # ========================================================================
    # Compute dimensionless strain measures:
    #   - Deviatoric strain: measures shape distortion (I_C - 2)/2
    #   - Volumetric strain: measures area change |J - 1|
    #
    # At rest: I_C = 2, J = 1 → both strains = 0
    #
    # Physically-motivated weighting based on material stiffness ratio:
    #   w_dev = k_mu / (k_mu + k_lambda)
    #   w_vol = k_lambda / (k_mu + k_lambda)
    #
    # Rationale:
    #   - Stiffer materials (higher k) contribute more resistance to that deformation mode
    #   - Weight each strain component by its material's relative stiffness
    #   - For soft materials (k_mu ≈ k_lambda): weights ≈ 0.5, 0.5 (equal)
    #   - For shear-dominant (k_mu >> k_lambda): weights ≈ 1.0, 0.0 (shape matters more)
    #   - For volume-preserving (k_lambda >> k_mu): weights ≈ 0.0, 1.0 (area matters more)
    #
    # Physical interpretation of SIGNED strain (like springs):
    #   total_strain > 0 : expansion/tension
    #   total_strain < 0 : compression
    #   total_strain = 0 : at rest (no deformation)
    #
    # Use SIGNED volumetric strain (J - 1) as primary metric:
    #   J > 1 → expansion → positive strain
    #   J < 1 → compression → negative strain
    #   J = 1 → at rest → zero strain
    #
    # Deviatoric strain (shape change) is signed by the volumetric direction
    deviatoric_strain = (Ic - 2.0) / 2.0  # Signed: positive = stretched, negative = compressed
    volumetric_strain = J - 1.0           # Signed: positive = expansion, negative = compression
    
    # Compute material-dependent weights (before area scaling was applied)
    # Use original k_mu, k_lambda values (divide out the area scaling)
    k_mu_orig = k_mu / rest_area
    k_lambda_orig = k_lambda / rest_area
    total_stiffness = k_mu_orig + k_lambda_orig
    
    # Normalized weights (sum to 1.0)
    w_deviatoric = k_mu_orig / total_stiffness
    w_volumetric = k_lambda_orig / total_stiffness
    
    # Material-dependent combined strain (SIGNED)
    total_strain = w_deviatoric * deviatoric_strain + w_volumetric * volumetric_strain
    
    # Store raw strain
    tri_strains[tid] = total_strain
    
    # Normalize strain to [-1, 1] range (branchless for GPU efficiency)
    # Use max to avoid division by zero, then clamp result
    safe_scale = wp.max(strain_scale, 1e-8)
    
    # Dead-zone: ignore strains with magnitude below noise threshold (1e-5)
    noise_threshold = 1e-5
    effective_strain = wp.where(wp.abs(total_strain) < noise_threshold, 0.0, total_strain)
    
    normalized = wp.clamp(effective_strain / safe_scale, -1.0, 1.0)
    tri_strains_normalized[tid] = normalized
    
    # ========================================================================
    # STEP 5: Apply Forces to Global Force Array
    # ========================================================================
    # Use atomic operations because multiple triangles can share vertices
    # Forces are accumulated additively across all elements
    #
    # Note: We use atomic_sub (subtract) because in the implicit solver,
    # forces are derived as F = -∇E (negative gradient of potential energy)
    # This ensures energy minimization and stable equilibrium
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.kernel
def eval_springs_2d(
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
    Evaluate spring forces for implicit solver.
    
    Identical to semi-implicit version but used for force evaluation in implicit method.
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
    
    if l < 1.0e-6:
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
    safe_scale = wp.max(strain_scale, 1e-8)
    normalized = wp.clamp(raw_strain / safe_scale, -1.0, 1.0)
    spring_strains_normalized[tid] = normalized

    # Spring force with damping
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def build_system_matrix_sparse_2d(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat22f),
    indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    mass: float,
    Minv: float
):
    """
    Build sparse system matrix for implicit integration (2D version).
    
    System matrix: A = M - h*D - h^2*K
    where M is mass matrix, D is damping, K is stiffness, h is timestep
    
    Each spring contributes 4 2x2 blocks:
        - (i,i): Mass and stiffness for particle i
        - (j,j): Mass and stiffness for particle j  
        - (i,j): Off-diagonal coupling
        - (j,i): Off-diagonal coupling (symmetric)
    
    Adapted from soft/kernels.py:build_system_matrix_sparse_kernel (3D -> 2D)
    """
    tid = wp.tid()
    
    i = indices[tid * 2 + 0]
    j = indices[tid * 2 + 1]
    
    # Get spring properties
    k = spring_stiffness[tid]
    d = spring_damping[tid]
    
    # Pre-compute common terms
    dt2 = dt * dt
    dt2_k = dt2 * k
    dt_d_Minv = dt * d * Minv
    
    # Diagonal blocks: M + h*D*M^-1 - h^2*K
    # For 2D, this is a 2x2 identity scaled by the scalar coefficient
    diag_coeff = mass + dt_d_Minv - dt2_k
    block_ii = wp.mat22f(
        diag_coeff, 0.0,
        0.0, diag_coeff
    )
    block_jj = wp.mat22f(
        diag_coeff, 0.0,
        0.0, diag_coeff
    )
    
    # Off-diagonal blocks: h^2*K
    offdiag_coeff = dt2_k
    block_ij = wp.mat22f(
        offdiag_coeff, 0.0,
        0.0, offdiag_coeff
    )
    
    # Store blocks (4 blocks per spring)
    block_idx = tid * 4
    
    # (i,i) block
    rows[block_idx] = i
    cols[block_idx] = i
    values[block_idx] = block_ii
    
    # (j,j) block
    rows[block_idx + 1] = j
    cols[block_idx + 1] = j
    values[block_idx + 1] = block_jj
    
    # (i,j) block
    rows[block_idx + 2] = i
    cols[block_idx + 2] = j
    values[block_idx + 2] = block_ij
    
    # (j,i) block
    rows[block_idx + 3] = j
    cols[block_idx + 3] = i
    values[block_idx + 3] = block_ij


@wp.kernel
def update_state_2d(
    dv: wp.array(dtype=wp.vec2),
    dt: float,
    positions_in: wp.array(dtype=wp.vec2),
    velocities_in: wp.array(dtype=wp.vec2),
    positions_out: wp.array(dtype=wp.vec2),
    velocities_out: wp.array(dtype=wp.vec2)
):
    """
    Update state after implicit solve.
    
    Given velocity change dv from linear system solve:
        v_{n+1} = v_n + dv
        x_{n+1} = x_n + v_{n+1} * dt
    """
    tid = wp.tid()
    
    vel = velocities_in[tid] + dv[tid]
    positions_out[tid] = positions_in[tid] + vel * dt
    velocities_out[tid] = vel


@wp.kernel
def eval_gravity_2d(
    gravity_force: wp.vec2,
    forces: wp.array(dtype=wp.vec2),
):
    """Apply gravity force to all particles."""
    tid = wp.tid()
    forces[tid] = gravity_force


@wp.kernel
def apply_boundary_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    boxsize: float,
):
    """
    Apply reflective boundary conditions in 2D.
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
    ground_y: float,
    direction: wp.vec2,  # Locomotion direction (normalized 2D vector)
):
    """
    Apply boundary conditions with DIRECTION-AWARE ratchet friction.
    
    Ratchet mechanism for any direction:
        - Forward (along direction): NO friction - slides freely
        - Backward (against direction): COMPLETE STOP - ratchet
    
    This enables locomotion in ANY 2D direction with symmetric radial forces.
    """
    tid = wp.tid()
    
    pos = x[tid]
    vel = v[tid]
    
    # X dimension (walls) - reflective
    if pos[0] < 0.0:
        pos = wp.vec2(-pos[0], pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    elif pos[0] > boxsize:
        pos = wp.vec2(boxsize - (pos[0] - boxsize), pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    
    # Y dimension - GROUND with DIRECTION-AWARE ratchet friction
    if pos[1] < ground_y:
        pos = wp.vec2(pos[0], ground_y - (pos[1] - ground_y))
        
        # Project velocity onto locomotion direction: v_forward = dot(vel, direction)
        v_forward = wp.dot(vel, direction)
        
        if v_forward > 0.0:
            # Moving forward: NO friction - slides freely
            vx_new = vel[0]
            vy_new = -vel[1]
        else:
            # Moving backward: STOP motion in direction
            vx_new = vel[0] - v_forward * direction[0]
            vy_new = -vel[1] - v_forward * direction[1]
        
        vel = wp.vec2(vx_new, vy_new)
    elif pos[1] > boxsize:
        pos = wp.vec2(pos[0], boxsize - (pos[1] - boxsize))
        vel = wp.vec2(vel[0], -vel[1])
    
    x[tid] = pos
    v[tid] = vel


# ============================================================================
# High-level wrapper functions
# ============================================================================

def eval_spring_forces_2d(model, state, particle_f: wp.array, spring_strains: wp.array = None):
    """Evaluate spring forces and compute normalized strains (wrapper function)."""
    if model.spring_count > 0:
        # Use model's strain arrays
        if spring_strains is None:
            spring_strains = model.spring_strains
        
        # Get current strain scale for normalization
        strain_scale = model.spring_strain_scale.numpy()[0]
        
        wp.launch(
            kernel=eval_springs_2d,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.spring_indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                particle_f,
                spring_strains,
                model.spring_strains_normalized,
                strain_scale,
            ],
            device=model.device,
        )


def eval_triangle_fem_forces_2d(model, state, particle_f: wp.array):
    """Evaluate FEM forces for triangular elements and compute normalized strains."""
    if hasattr(model, 'tri_count') and model.tri_count > 0:
        # Get current FEM strain scale for normalization
        fem_strain_scale = model.fem_strain_scale.numpy()[0]
        
        wp.launch(
            kernel=eval_triangles_fem_2d,
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

