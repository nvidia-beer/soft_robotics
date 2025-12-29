# Integration Methods: Semi-Implicit vs Implicit

## Mathematical Formulation

### Semi-Implicit Euler (Symplectic Euler)

A **first-order symplectic integrator** that evaluates forces explicitly but integrates velocity before position.

**Algorithm:**
```
1. Evaluate forces: f_n = F(x_n, v_n)
2. Half kick:       v_{n+1/2} = v_n + (f_n/m + g) * dt/2
3. Drift:           x_{n+1} = x_n + v_{n+1/2} * dt
4. Evaluate forces: f_{n+1} = F(x_{n+1}, v_{n+1/2})
5. Final kick:      v_{n+1} = v_{n+1/2} + (f_{n+1}/m + g) * dt/2
```

**Continuous form:**
\[
\begin{align}
\mathbf{v}_{n+1} &= \mathbf{v}_n + \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}_n, \mathbf{v}_n) \Delta t \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \mathbf{v}_{n+1} \Delta t
\end{align}
\]

**Properties:**
- ✅ **Symplectic**: Preserves phase space volume, no artificial energy drift
- ⚠️ **Conditionally stable**: \( \Delta t < 2\sqrt{m/k} \) for spring systems
- ⚠️ **Explicit**: No matrix inversion required (fast per-step, small steps needed)

---

### Implicit Euler (Backward Euler)

A **first-order implicit integrator** that solves a linear system to evaluate forces at the next time step.

**Algorithm:**
```
1. Build system matrix: A = M - h*D - h²*K
   where K = ∂f/∂x (stiffness), D = ∂f/∂v (damping)
2. Evaluate forces: f_n = F(x_n, v_n)
3. Solve linear system: A * Δv = f_n * dt
4. Update velocity: v_{n+1} = v_n + Δv
5. Update position: x_{n+1} = x_n + v_{n+1} * dt
```

**Continuous form (linearized):**
\[
\begin{align}
\left(\mathbf{M} - \Delta t \mathbf{D} - \Delta t^2 \mathbf{K}\right) \Delta \mathbf{v} &= \mathbf{f}_n \Delta t \\
\mathbf{v}_{n+1} &= \mathbf{v}_n + \Delta \mathbf{v} \\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \mathbf{v}_{n+1} \Delta t
\end{align}
\]

where:
- \(\mathbf{M}\): Mass matrix (diagonal)
- \(\mathbf{D} = -\frac{\partial \mathbf{f}}{\partial \mathbf{v}}\): Damping matrix (negative Jacobian w.r.t. velocity)
- \(\mathbf{K} = -\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\): Stiffness matrix (negative Jacobian w.r.t. position)

**Properties:**
- ✅ **Unconditionally stable**: Any \(\Delta t\) works (no CFL condition)
- ⚠️ **Not symplectic**: Artificially dissipates energy (stable but non-physical)
- ⚠️ **Implicit**: Requires sparse linear system solve (expensive per-step, large steps possible)

---

## Comparison Table

| Property | Semi-Implicit | Implicit |
|----------|--------------|----------|
| **Order** | First-order | First-order |
| **Stability** | Conditional: \(\Delta t < 2\sqrt{m/k}\) | Unconditional |
| **Energy** | Conserved (symplectic) | Dissipated |
| **Cost/step** | Low (explicit) | High (linear solve) |
| **Matrix assembly** | None | Once (linear springs) |
| **Iteration** | None | 3-10 CG/BiCGStab iterations |
| **Best for** | Small dt, rigid constraints | Large dt, soft deformables |

---

## Stability Analysis

### Semi-Implicit (Spring System)

For a 1D spring \(f = -kx - c\dot{x}\), the stability condition is:

\[
\Delta t < \frac{2}{\omega_0}, \quad \omega_0 = \sqrt{\frac{k}{m}}
\]

**Example:** For \(k=100\), \(m=1\): \(\Delta t < 0.02\) seconds.

### Implicit (Spring System)

The amplification matrix has eigenvalues \(\lambda\) satisfying:

\[
|\lambda| \leq 1 \quad \forall \Delta t > 0
\]

Thus **unconditionally stable** for any time step.

---

## Implementation Notes

### Semi-Implicit
- Forces evaluated **twice per step** (at \(t_n\) and \(t_{n+1}\))
- No matrix construction or linear solve
- Suitable for real-time simulations with small \(\Delta t\)

### Implicit
- System matrix \(\mathbf{A}\) assembled **once** (constant for linear springs)
- Sparse BSR (Block Sparse Row) format with 2×2 blocks for 2D
- Iterative solvers: BiCGStab (recommended), CG, GMRES, CR
- Preconditioned for faster convergence (Jacobi/diagonal)
- Suitable for offline simulations, soft bodies, large \(\Delta t\)

---

## When to Use Each Method

**Use Semi-Implicit if:**
- Real-time performance critical
- Stiff springs but fast timestep acceptable
- Energy conservation important (e.g., orbital mechanics)

**Use Implicit if:**
- Very stiff systems (large \(k\))
- Large time steps needed
- Stability more important than exact energy conservation
- Soft deformable bodies (FEM with Neo-Hookean materials)






















