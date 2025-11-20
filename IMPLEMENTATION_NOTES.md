# Implementation Notes

This document describes the Python implementation and its differences from the original MATLAB version.

## Overview

This Python implementation translates the MATLAB MPC spring-mass example to Python, maintaining the same physical system and control objectives while adapting the implementation to Python's scientific computing ecosystem.

## Key Differences from MATLAB Implementation

### 1. Optimization Solver

**MATLAB:**
- Uses YALMIP optimization modeling toolbox
- Can use various solvers (linprog, quadprog, etc.)
- Convex optimization with explicit linear/quadratic programming formulation

**Python:**
- Uses SciPy's `minimize` with SLSQP (Sequential Least Squares Programming)
- Nonlinear constraint handling via `NonlinearConstraint`
- Trade-off: Less efficient than specialized LP/QP solvers, but more flexible and doesn't require additional dependencies

**Potential improvement:** Could use CVXPY with specialized solvers (OSQP, ECOS, etc.) for better performance, but this adds dependencies.

### 2. System Integration

**MATLAB:**
- Uses `ode45` (Dormand-Prince Runge-Kutta method)
- Explicit control input functions passed to ODE solver

**Python:**
- Uses `scipy.integrate.solve_ivp` with RK45 method (similar to ode45)
- Same mathematical approach, Python-style API

### 3. Code Structure

**MATLAB:**
- Separate function files for different components
- Script-based driver with sections
- Global parameters in struct

**Python:**
- Object-oriented design with `SpringMassMPC` class
- Encapsulated state and parameters
- More modular and reusable

### 4. Discretization

Both implementations use the same trapezoidal (Crank-Nicolson) discretization:

```
x(k+1) = x(k) + 0.5 * (v(k) + v(k+1)) * dt
v(k+1) = v(k) + 0.5 * (a(k) + a(k+1)) * dt
```

### 5. MPC Formulation

**Objective Function:**
Both minimize the same cost:
```
J = R * ||x||_1 + Q * ||u||_1
```

**Constraints:**
- Initial conditions: `x(0) = x0`, `v(0) = v0`
- Dynamics: Position and velocity updates
- Force balance: `m*a + 2*c*v + spring_forces = u` (for controlled masses)
- Control bounds: `|u| <= u_max`

### 6. Nonlinear Springs

**MATLAB:**
- Nonlinearity included in ODE simulation
- MPC uses linearized model (assumes linear springs)

**Python:**
- Same approach: nonlinear simulation, linear MPC
- `is_linear` flag controls which model to use
- Both `h(r) = k*r` and `h(r) = k*r - k_nl*r^3` supported

**Note:** To fully exploit nonlinear spring models, the MPC formulation would need to include the nonlinear terms in the optimization (requires nonlinear MPC).

## Performance Considerations

### Computational Speed

The MATLAB implementation with YALMIP and linprog is likely faster because:
1. Specialized LP/QP solvers are highly optimized
2. MATLAB's matrix operations are heavily optimized
3. YALMIP generates efficient problem formulations

The Python implementation trades some speed for:
1. No additional dependencies beyond SciPy
2. More flexible constraint handling
3. Easier to extend and modify

### Recommended Improvements for Speed

1. **Use CVXPY:** Replace SLSQP with CVXPY + OSQP for much faster solving
2. **Reduce horizon:** Use smaller prediction horizon N (e.g., 30-50 instead of 100)
3. **Warm-start:** Use previous solution as initial guess
4. **Sparse matrices:** For large problems, use sparse matrix formulations
5. **JIT compilation:** Use Numba for critical loops

## Usage Patterns

### MATLAB Pattern
```matlab
% Create solver
[MpcSolver, time_mpc] = mpc_linear_opt(p);

% Solve in loop
for iter = 1:500
    [sol, errorcode] = MpcSolver({x0, v0});
    % ... use solution ...
end
```

### Python Pattern
```python
# Create controller
mpc = SpringMassMPC(M=7, m=1.0, k=5.0, c=0.1)

# Solve in loop
for i in range(500):
    x_next, v_next, u_applied, x_pred = mpc.step(x_current, v_current)
    # ... use solution ...
```

## Testing

The Python implementation includes:
- `test_basic.py`: Unit tests for core functionality
- `example_simple.py`: Simple 5-mass example
- `driverSpringMassMPC.py`: Full simulation matching MATLAB outputs

## Validation

To validate against MATLAB:
1. Use same initial conditions
2. Compare uncontrolled trajectories (should be nearly identical)
3. Compare controlled trajectories (similar behavior, may differ due to solver differences)
4. Check that control constraints are satisfied
5. Verify that controller stabilizes the system

## Extension Points

The class-based design makes it easy to extend:

1. **Add state estimation:**
```python
class SpringMassMPCWithEstimator(SpringMassMPC):
    def estimate_state(self, measurements):
        # Add Kalman filter or observer
        pass
```

2. **Add parameter adaptation:**
```python
class AdaptiveSpringMassMPC(SpringMassMPC):
    def update_parameters(self, observations):
        # Learn k, m, c from data
        pass
```

3. **Add nonlinear MPC:**
```python
class NonlinearSpringMassMPC(SpringMassMPC):
    def mpc_constraints(self, decision_vars, x0, v0):
        # Include nonlinear spring terms in optimization
        pass
```

## Known Limitations

1. **Solver speed:** SLSQP is slower than specialized LP solvers
2. **No warm-start:** Each solve starts from scratch
3. **No parallelization:** Could parallelize multiple scenarios
4. **Memory usage:** Stores full trajectories (could be optimized)
5. **No real-time guarantees:** Solve time can vary

## Future Work

1. Implement with CVXPY for better performance
2. Add Kalman filter for state estimation
3. Implement nonlinear MPC variant
4. Add parameter learning/identification
5. Create interactive visualization
6. Add constraint tightening for robust MPC
7. Implement distributed MPC for very large systems

## References

- **Original MATLAB:** https://github.com/bjarkegosvig/mpc-spring-mass-example
- **YALMIP:** https://yalmip.github.io/
- **SciPy Optimize:** https://docs.scipy.org/doc/scipy/reference/optimize.html
- **CVXPY:** https://www.cvxpy.org/ (recommended for faster solving)

