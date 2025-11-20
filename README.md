# MPC for Spring-Mass System - Python Implementation

A Python implementation of Model Predictive Control (MPC) for a nonlinear spring-mass system. This project is a Python translation of the [mpc-spring-mass-example](https://github.com/bjarkegosvig/mpc-spring-mass-example) MATLAB implementation.

## System Description

The system consists of `M` masses (`m_1`, `m_2`, ..., `m_M`) connected by springs with velocity-dependent friction. The system is controlled by applying forces `u1` and `u2` to the first and last masses. The objective is to keep the masses as close to their equilibrium positions (x=0) as possible while minimizing control effort.

```
|   k     _______    k     _______         k     _______    k    |
|__/\/\__|       |__/\/\__|       |_ ... _/\/\__|       |__/\/\__|
|        |  m_1  |        |  m_2  |             |  m_M  |        |
|  u1 -->|_______|        |_______|             |_______|--> u2  |   x
============(c)==============(c)====== ... ========(c)================== -->
```

### System Dynamics

The dynamics are governed by:

```
m*x_i'' + 2*c*x_i' + h(x_i - x_{i-1}) + h(x_i - x_{i+1}) = 0, for i=1..M
```

where:
- `x_i` is the position of mass `i`
- `x_0 = x_{M+1} = 0` (boundary conditions)
- `x_i'` and `x_i''` are velocity and acceleration
- `h(r) = k*r` for linear springs
- `h(r) = k*r - k_nl*r^3` for nonlinear springs

### Control Inputs

- `u1`: Force applied to the first mass (left side)
- `u2`: Force applied to the last mass (right side)
- Control inputs are bounded: `|u1|, |u2| <= u_max`

## Features

- **Model Predictive Control**: Optimization-based control with prediction horizon
- **Linear and Nonlinear Springs**: Support for both linear and nonlinear spring models
- **Noise Handling**: Control signals can include noise to simulate real-world conditions
- **Visualization**: Comprehensive plotting of controlled vs uncontrolled behavior
- **Flexible Parameters**: Easy configuration of system and MPC parameters

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

### Setup

1. Clone or navigate to this repository:
```bash
cd /home/beer/NBEL/soft_mpc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from SpringMassMPC import SpringMassMPC
import numpy as np

# Create MPC controller
mpc = SpringMassMPC(
    M=7,           # 7 masses
    m=1.0,         # 1 kg per mass
    k=5.0,         # Spring stiffness
    c=0.1,         # Friction coefficient
    u_max=5.0,     # Max control force
    dt=0.1,        # Time step
    N=50,          # Prediction horizon
    Q=1.0,         # Control weight
    R=50.0,        # State weight
    is_linear=True # Linear springs
)

# Initial conditions
x0 = np.array([-1.0, 3.0, 1.5, -4.0, 0.3, -0.5, -0.3])
v0 = np.zeros(7)

# Run one control step
x_next, v_next, u_applied, x_pred = mpc.step(x0, v0, noise_magnitude=0.5)
```

### Run Complete Simulation

Run the driver script to see a full simulation with visualizations:

```bash
python driverSpringMassMPC.py
```

This will:
1. Simulate the uncontrolled system
2. Simulate the MPC-controlled system
3. Generate three plots:
   - Controlled vs uncontrolled trajectories
   - Control inputs over time
   - Controlled trajectory with MPC predictions

## File Structure

```
soft_mpc/
├── SpringMassMPC.py          # Main MPC controller class
├── driverSpringMassMPC.py    # Example driver code
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Algorithm Details

### MPC Formulation

The MPC problem is formulated as:

**Minimize:**
```
J = R * ||x||_1 + Q * ||u||_1
```

**Subject to:**
- System dynamics (discretized using trapezoidal rule)
- Control constraints: `-u_max <= u <= u_max`
- Initial conditions: `x(0) = x0`, `v(0) = v0`

where:
- `x` is the position trajectory over the prediction horizon
- `u` is the control input sequence
- `R` weights the position error
- `Q` weights the control effort

### Discretization

The continuous-time dynamics are discretized using the trapezoidal (Crank-Nicolson) method:

```
x(k+1) = x(k) + 0.5 * (v(k) + v(k+1)) * dt
v(k+1) = v(k) + 0.5 * (a(k) + a(k+1)) * dt
```

### Optimization

The optimization problem is solved using Sequential Least Squares Programming (SLSQP) from SciPy.

## Parameters

### System Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Number of masses | 7 |
| `m` | Mass of each element (kg) | 1.0 |
| `k` | Linear spring coefficient | 5.0 |
| `c` | Friction coefficient | 0.1 |
| `k_nl` | Nonlinear spring coefficient | 0.01 |
| `u_max` | Maximum control force | 5.0 |

### MPC Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dt` | Time step (s) | 0.1 |
| `N` | Prediction horizon (steps) | 100 |
| `Q` | Control weight | 1.0 |
| `R` | State weight | 50.0 |
| `is_linear` | Use linear springs | True |

## Results

The controller successfully stabilizes the spring-mass system:

- **Uncontrolled**: Masses oscillate indefinitely with slow decay due to friction
- **Controlled**: Masses quickly converge to equilibrium positions (x=0)
- **MPC Predictions**: Show how the controller anticipates future behavior

Example output plots are saved as:
- `controlled_vs_uncontrolled.png`
- `control_inputs.png`
- `controlled_with_predictions.png`

## Potential Improvements

1. **Nonlinear MPC**: Incorporate nonlinear spring model directly in the MPC formulation
2. **State Estimation**: Add an observer/Kalman filter for noisy measurements
3. **Computational Speed**: Use sparse matrices and more efficient solvers (e.g., CVXPY with specialized solvers)
4. **Real-time Implementation**: Add computation time delays to simulate realistic control loops
5. **Parameter Learning**: Learn system parameters (k, m, c) from observations

## References

- Original MATLAB implementation: [mpc-spring-mass-example](https://github.com/bjarkegosvig/mpc-spring-mass-example)
- Python MPC tutorial: [Model Predictive Control Tutorial](https://aleksandarhaber.com/model-predictive-control-mpc-tutorial-1-unconstrained-formulation-derivation-and-implementation-in-python-from-scratch/)

## License

MIT License

## Author

Python implementation by: MPC Spring-Mass Team  
Date: November 2025

