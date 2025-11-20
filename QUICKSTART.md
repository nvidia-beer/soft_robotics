# Quick Start Guide

Get started with the Spring-Mass MPC implementation in 5 minutes!

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Navigate to the directory:**
   ```bash
   cd /home/beer/NBEL/soft_mpc
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - NumPy (numerical computing)
   - SciPy (optimization and integration)
   - Matplotlib (plotting)

## Verify Installation

Run the test suite to ensure everything is working:

```bash
python3 test_basic.py
```

You should see all 6 tests pass:
```
============================================================
Running Basic Tests for SpringMassMPC
============================================================

Test 1: Initialization... ✓ PASSED
Test 2: Spring force... ✓ PASSED
Test 3: System dynamics... ✓ PASSED
Test 4: Uncontrolled simulation... ✓ PASSED
Test 5: MPC solve (small problem)... ✓ PASSED
          Initial magnitude: 1.000
          Final magnitude:   0.202
Test 6: Control step... ✓ PASSED

============================================================
Results: 6/6 tests passed
✓ All tests passed!
============================================================
```

## Run Examples

### Simple Example

Run a basic 5-mass system example:

```bash
python3 example_simple.py
```

This will:
- Create a 5-mass MPC controller
- Solve one optimization problem
- Run 10 control steps
- Generate a plot showing predicted trajectories

**Expected output:**
```
============================================================
Simple Spring-Mass MPC Example
============================================================

1. Creating MPC controller...
   ✓ Controller created

2. Setting initial conditions...
   Initial positions: [ 2.  -1.5  1.  -0.5  0.8]
   Initial velocities: [0. 0. 0. 0. 0.]

3. Solving MPC optimization problem...
   ✓ Optimization succeeded
   ...
```

### Full Simulation

Run a complete simulation with 7 masses:

```bash
python3 driverSpringMassMPC.py
```

**Note:** This will take several minutes as it runs 500 MPC iterations. You'll see progress updates:
```
Simulating controlled system with MPC...
  Step 0/500
  Step 10/500
  Step 20/500
  ...
```

**Output:**
Three plots will be generated:
1. `controlled_vs_uncontrolled.png` - Comparison of system behavior
2. `control_inputs.png` - Control forces over time
3. `controlled_with_predictions.png` - Controlled trajectory with MPC predictions

## Understanding the Results

### What the Controller Does

The MPC controller:
1. **Predicts** the future behavior of the spring-mass system
2. **Optimizes** control inputs to minimize position deviations and control effort
3. **Applies** the first control action
4. **Repeats** at each time step

### Key Observations

**Uncontrolled System:**
- Masses oscillate indefinitely
- Slow decay due to friction
- Can take 100+ seconds to settle

**Controlled System:**
- Masses quickly return to equilibrium (x=0)
- Typically settles in 10-20 seconds
- Active damping via control inputs

### Performance Metrics

From the simple example, you should see something like:
```
6. Results:
   Initial max position: 2.000
   Final max position:   0.157
   Reduction: 92.1%
```

After 10 steps (1 second), the controller reduces position errors by ~90%!

## Quick Code Example

Here's the minimal code to use the controller:

```python
import numpy as np
from SpringMassMPC import SpringMassMPC

# Create controller
mpc = SpringMassMPC(M=5, u_max=5.0, dt=0.1, N=30)

# Set initial state (displaced masses)
x = np.array([1.0, -0.5, 0.5, -0.3, 0.2])
v = np.zeros(5)

# Run 10 control steps
for i in range(10):
    x, v, u, x_pred = mpc.step(x, v, noise_magnitude=0.0)
    print(f"Step {i}: max position = {np.max(np.abs(x)):.3f}")
```

## Parameter Tuning

Key parameters to adjust:

| Parameter | What it does | Increase to... | Decrease to... |
|-----------|-------------|----------------|----------------|
| `R` | Position weight | Make controller more aggressive | Allow larger deviations |
| `Q` | Control weight | Reduce control effort | Allow larger control |
| `N` | Prediction horizon | Look further ahead (slower) | Faster computation |
| `dt` | Time step | Slower dynamics | Faster response |
| `u_max` | Max force | Stronger control | Limit actuator force |

### Example: More Aggressive Control

```python
mpc = SpringMassMPC(M=5, R=100.0, Q=0.1, u_max=10.0)  # High R, low Q, high u_max
```

### Example: Gentler Control

```python
mpc = SpringMassMPC(M=5, R=10.0, Q=10.0, u_max=2.0)   # Low R, high Q, low u_max
```

## Common Issues

### 1. Optimization doesn't converge

**Problem:** `Optimization warning: Iteration limit reached`

**Solutions:**
- Reduce prediction horizon `N` (try 30-50)
- Increase `maxiter` in `solve_mpc` method
- Check that initial conditions are reasonable
- Reduce system size (fewer masses)

### 2. Slow computation

**Problem:** Each MPC step takes too long

**Solutions:**
- Reduce `N` (prediction horizon)
- Use fewer masses `M`
- Consider using CVXPY with OSQP solver (see IMPLEMENTATION_NOTES.md)

### 3. Controller doesn't stabilize system

**Problem:** Positions don't converge to zero

**Solutions:**
- Increase `R` (position weight)
- Increase `u_max` (control force limit)
- Check that system parameters are physical (positive m, k, c)
- Verify control inputs are being applied correctly

## Next Steps

1. **Read the full README.md** for detailed system description
2. **Check IMPLEMENTATION_NOTES.md** for technical details and comparisons with MATLAB
3. **Modify parameters** in `example_simple.py` and experiment
4. **Add your own features** - the class-based design makes extension easy

## Getting Help

If you encounter issues:

1. Run `test_basic.py` to ensure installation is correct
2. Check that all dependencies are installed: `pip list | grep -E "(numpy|scipy|matplotlib)"`
3. Review error messages carefully - they often indicate parameter issues
4. Try the simple example first before the full simulation

## Performance Expectations

On a typical modern laptop:

| Task | Time | Notes |
|------|------|-------|
| One MPC solve (N=30, M=5) | ~0.5-2 sec | Depends on initial conditions |
| One MPC solve (N=50, M=7) | ~2-5 sec | Larger problem |
| Simple example | ~30 sec | 10 steps with small problem |
| Full simulation | ~10-30 min | 500 steps with larger problem |

**Tip:** Start with small problems (M=3-5, N=20-30) for testing, then scale up.

---

## Summary

You now know how to:
- ✓ Install and test the implementation
- ✓ Run examples
- ✓ Use the controller in your own code
- ✓ Tune parameters
- ✓ Troubleshoot common issues

**Next:** Try modifying `example_simple.py` to experiment with different configurations!

