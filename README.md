# MPC for Spring-Mass System

A Python implementation of Model Predictive Control (MPC) for a nonlinear spring-mass system.

## System Description

The system consists of `M` masses connected by springs with velocity-dependent friction. Control forces `u1` and `u2` are applied to the first and last masses to stabilize the system.

```
|   k     _______    k     _______         k     _______    k    |
|__/\/\__|       |__/\/\__|       |_ ... _/\/\__|       |__/\/\__|
|        |  m_1  |        |  m_2  |             |  m_M  |        |
|  u1 -->|_______|        |_______|             |_______|--> u2  |
```

### System Dynamics

```
m*x_i'' + 2*c*x_i' + h(x_i - x_{i-1}) + h(x_i - x_{i+1}) = 0
```

where `h(r) = k*r` for linear springs or `h(r) = k*r - k_nl*r^3` for nonlinear springs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from soft_mpc import SpringMassMPC
import numpy as np

# Create MPC controller
mpc = SpringMassMPC(
    M=5,           # 5 masses
    m=1.0,         # Mass (kg)
    k=5.0,         # Spring stiffness
    c=0.1,         # Friction coefficient
    u_max=5.0,     # Max control force
    dt=0.1,        # Time step
    N=50           # Prediction horizon
)

# Initial conditions
x0 = np.array([1.0, -0.5, 0.5, -0.3, 0.2])
v0 = np.zeros(5)

# Run control loop
for i in range(100):
    x_next, v_next, u_applied, x_pred = mpc.step(x0, v0)
    x0, v0 = x_next, v_next
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `M` | Number of masses | 7 |
| `m` | Mass of each element (kg) | 1.0 |
| `k` | Spring stiffness | 5.0 |
| `c` | Friction coefficient | 0.1 |
| `u_max` | Maximum control force | 5.0 |
| `dt` | Time step (s) | 0.1 |
| `N` | Prediction horizon | 100 |
| `Q` | Control weight | 1.0 |
| `R` | Position weight | 50.0 |

## MPC Formulation

**Objective:**
```
minimize: R * ||x||_1 + Q * ||u||_1
```

**Subject to:**
- System dynamics (discretized using trapezoidal rule)
- Control constraints: `-u_max ≤ u ≤ u_max`
- Initial conditions: `x(0) = x0`, `v(0) = v0`

## Reference

Based on the MATLAB implementation:  
https://github.com/bjarkegosvig/mpc-spring-mass-example

## License

MIT License - see LICENSE file for details.
