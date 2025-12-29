# Trajectory Tracking with SNN-Enhanced MPC

This project implements trajectory tracking on a spring FEM grid, demonstrating the benefit of SNN adaptation for continuous control tasks.

**SIMPLIFIED**: External forces applied ONLY to the CENTER GROUP.
Requires even grid size (4x4, 6x6, etc.)

## Why Trajectory Tracking?

Unlike stabilization (where classical MPC wins), trajectory tracking creates **persistent prediction errors** that SNNs can learn from:

| Task | Prediction Errors | SNN Benefit |
|------|-------------------|-------------|
| Stabilization | Transient, disappear when settled | ❌ None |
| Trajectory Tracking | Continuous, never disappear | ✅ Significant |

This follows the approach from [Halaly & Tsur 2024](https://doi.org/10.1088/2634-4386/ad4209) - "Continuous adaptive nonlinear model predictive control using spiking neural networks".

## Structural Mismatch (by design)

- **Simulation**: FEM triangles (Neo-Hookean hyperelastic - nonlinear, accurate)
- **MPC Model**: Simple spring-damper (linear, fast approximation)

This creates a **natural mismatch** that the SNN can learn to correct - no artificial parameter errors needed!

## Grid Structure

```
4x4 Grid with 9 overlapping 2x2 groups (3x3 arrangement):

    0──1──2──3         Groups (3x3):
    │╲/│╲/│╲/│         [0][1][2]
    4──5──6──7         [3][4][5]  <- Group 4 is CENTER
    │╲/│╲/│╲/│         [6][7][8]
    8──9─10─11
    │╲/│╲/│╲/│
   12─13─14─15

CENTER GROUP (4) contains particles [5, 6, 9, 10]
  → ONLY these 4 particles receive external forces!

Each group (2x2 cell):
    ┌───────┐
    │ ●   ● │   4 particles
    │  ╲ ╱  │   2 FEM triangles (diagonal split)
    │  ╱ ╲  │   5 springs (4 edges + 1 diagonal)
    │ ●   ● │
    └───────┘

Why even grids only?
  - 3x3: 4 groups (2x2) - NO center group!
  - 4x4: 9 groups (3x3) - center is group 4 ✓
  - 6x6: 25 groups (5x5) - center is group 12 ✓
```

## Project Structure

```
trajectory_tracking/
├── controllers/              # Controller implementations
│   ├── __init__.py          # BaseController + exports
│   ├── mpc/__init__.py      # MPC (simplified centroid model)
│   ├── pid/__init__.py      # PID (model-free)
│   └── nengo/               # Nengo SNN controllers
│       ├── __init__.py
│       ├── base.py          # NengoControllerBase
│       ├── mpc.py           # NengoMPC (MPC + SNN)
│       └── pid.py           # NengoPID (NEF-based spiking PID)
├── tracking_env.py          # FEM grid environment
├── run_tracking.py          # Main script
├── run_tracking.sh          # Docker runner
└── README.md                # This file
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NxN FEM Grid (N even, N>=4)          │
│                                                          │
│   (N-1)×(N-1) groups, each with 4 particles             │
│   ONLY CENTER GROUP receives external forces            │
│                                                          │
│   Visualization:                                         │
│   - Pink circle: Center group centroid (tracks target)  │
│   - Black crosshair: Target position                    │
│   - Arrows: Control forces on center group particles    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              MPC Controller (SIMPLIFIED MODEL)           │
│                                                          │
│    Model: m*a = F_control - k*x - c*v                   │
│           (Simple spring-damper, linear)                 │
│                                                          │
│    Reality: FEM triangles (nonlinear!)                  │
│    → STRUCTURAL MISMATCH = SNN learning opportunity     │
└─────────────────────────────────────────────────────────┘
                         │
                         + SNN Correction
                         │
┌─────────────────────────────────────────────────────────┐
│              SNN Dynamics Adaptation                     │
│                                                          │
│    Training: prediction_error = simple_model - FEM      │
│    Output: force_correction                              │
│    Learning: PES (online, continuous)                    │
└─────────────────────────────────────────────────────────┘
```

## Controllers

### MPC (Classical)

- Uses **simplified centroid model** (spring-damper)
- Much faster than FEM for predictions
- Inherently wrong (doesn't match FEM reality)
- No adaptation capability

### NengoMPC (MPC + SNN)

- Composes classical MPC + SNN adaptation
- SNN learns to correct model prediction errors
- Uses PES (Prescribed Error Sensitivity) learning
- Continuous online learning during operation

### PID (Classical)

- Model-free proportional-integral-derivative controller
- Simple, robust, no learning

### NengoPID (SNN-PID)

- NEF-based spiking PID controller (Zaidel et al. 2021)
- Uses 4 ensembles: q(t), ei(t), ed(t), u(t)
- Optional PES feedforward learning from strain

## Running

```bash
# Interactive menu
./run_tracking.sh

# Direct run with specific controller
./run_tracking.sh --controller mpc         # Classical MPC
./run_tracking.sh --controller snn_mpc     # MPC + SNN
./run_tracking.sh --compare                # Compare MPC vs MPC+SNN

# Specify grid size (must be even >= 4)
./run_tracking.sh --controller mpc -N 6    # 6x6 grid

# Without visualization
./run_tracking.sh --controller mpc --no-display
```

## Expected Results

- **MPC**: Works okay but has persistent tracking error (simple model ≠ FEM)
- **MPC+SNN**: Learns to correct, tracking error decreases over time
- **Key metric**: Late-phase error should be lower for SNN

## GUI Elements

- **Grid**: NxN with FEM triangles and strain coloring
- **Pink circle**: Center group centroid (tracks target)
- **Black crosshair**: Target trajectory position
- **Arrows**: Control forces on center group particles only
- **Plots (right)**: X, Y position, XY trajectory, tracking error over time
