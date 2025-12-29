# Soft Body Trajectory Tracking with Spiking Neural Networks
## Project State Summary

---

## 1. System Overview

This project implements a **soft body trajectory tracking system** combining:
- GPU-accelerated physics simulation (Warp)
- Spring-mass-FEM deformable body
- Classical PID/MPC control
- Nengo spiking neural network visualization

---

## 2. Physical Simulation

### 2.1 Grid Structure

- **Particles**: NxN grid of mass points (configurable, minimum 2x2)
- **Springs**: Connect adjacent particles (horizontal, vertical, diagonal)
- **FEM Triangles**: Continuum mechanics elements for realistic deformation
- **Groups**: Overlapping 2x2 cells, each with a controllable centroid
  - 3x3 grid → 4 groups
  - 4x4 grid → 9 groups
  - 5x5 grid → 16 groups

### 2.2 Physics Engine

- **Framework**: Warp (NVIDIA GPU-accelerated simulation)
- **Solver**: Implicit integration
- **Time step**: 0.01 seconds
- **No gravity** (per design requirement)

---

## 3. Control System

### 3.1 Available Controllers

| Controller | Type | Status |
|------------|------|--------|
| PID | Classical proportional-integral-derivative | ✅ Working |
| MPC | Model predictive control | ✅ Working |
| SNN_PID | NEF-based spiking neural network PID | ✅ Working |

**SNN_PID** is based on [Zaidel et al., "Neuromorphic NEF-Based Inverse Kinematics and PID Control", Frontiers in Neurorobotics, 2021](https://doi.org/10.3389/fnbot.2021.631159). It implements:
- Neural integrator for the I term (recurrent connection)
- Neural differentiator for the D term (fast-slow approximation)
- All computation done with spiking neurons

### 3.2 PID Parameters (adjustable in real-time via Nengo GUI)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Kp | 200 | 0-500 | Proportional gain |
| Ki | 10 | 0-50 | Integral gain |
| Kd | 50 | 0-200 | Derivative gain |
| u_max | 500 | 100-1000 | Maximum control force |

### 3.3 Control Output

- **Dimension**: num_groups × 2 (x,y force per group centroid)
- **Applied to**: Group centroids, distributed to constituent particles

---

## 4. Trajectory Generation

### 4.1 Available Patterns (switchable in real-time)

| Pattern | Index | Motion |
|---------|-------|--------|
| Sinusoidal | 0 | Left-right oscillation |
| Circular | 1 | Clockwise circle |
| Figure-8 | 2 | Infinity pattern |

### 4.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Amplitude | 0.3 | Motion amplitude |
| Frequency | 0.2 Hz | Oscillation frequency |

---

## 5. Sensor Signals Available

### 5.1 Strain/Tension Data

| Signal | Source | Normalization |
|--------|--------|---------------|
| Spring strains | Individual spring deformation | [-1, 1] |
| FEM strains | Triangle element strain | [-1, 1] |
| Group average strain | Per 2x2 group | [-1, 1] |

### 5.2 State Information

| Signal | Description |
|--------|-------------|
| Group centroids | Current position of each group center |
| Group velocities | Current velocity of each group center |
| Target positions | Desired position from trajectory |
| Tracking error | Distance between target and current position |

---

## 6. Nengo SNN Layer

### 6.1 Current Function

The Nengo network provides **visualization only**:
- Strain signals → Rate-coded into spiking neurons → Spike raster display
- Control forces → Displayed as ensemble activity
- **No learning** - SNN does not influence control

### 6.2 Network Components

| Component | Neurons | Purpose |
|-----------|---------|---------|
| Strain ensembles | 50 per strain | Visualize internal stress |
| Group average ensembles | 100 per group | Aggregate strain per region |
| Force display ensembles | 100 per group | Visualize control output |

### 6.3 Nengo GUI Controls

| Slider | Function |
|--------|----------|
| Controller [0=OFF, 1=ON] | Enable/disable control |
| Type [0=PID, 1=MPC] | Select controller |
| PID Kp [0-500] | Proportional gain |
| PID Ki [0-50] | Integral gain |
| PID Kd [0-200] | Derivative gain |
| u_max [100-1000] | Maximum force |
| Traj [0-2] | Trajectory pattern |

---

## 7. Technical Constraints

### 7.1 Nengo GUI Limitation

Nengo GUI manages its own simulator. External controllers that create `nengo.Simulator` instances cause `StartedSimulatorException`. Therefore:
- NengoPID class (with internal PES learning) → **Cannot be used in GUI**
- NengoMPC class (with internal PES learning) → **Cannot be used in GUI**
- Classic PID/MPC (no internal simulator) → **Works in GUI**

### 7.2 Current Architecture Decision

Due to the above constraint, the GUI uses classic PID/MPC controllers. The original NengoPID/NengoMPC classes with PES learning exist but are not compatible with Nengo GUI operation.

---

## 8. Current Performance

With default PID settings (Kp=200, Ki=10, Kd=50):
- Tracking is stable
- Soft body follows trajectory accurately
- Real-time parameter adjustment works
- All trajectory patterns function correctly

---

## 9. File Structure

```
soft_robotics/trajectory_tracking/
├── snn_nengo_tracking_gui.py       # Main Nengo GUI application
├── snn_nengo_tracking_interface.py # SNN interface class  
├── tracking_env.py                 # Physics simulation environment
├── run_nengo_tracking.sh           # Launch script
├── controllers/
│   ├── __init__.py
│   ├── base.py                     # Base controller class
│   ├── pid/__init__.py             # Classic PID
│   ├── mpc/__init__.py             # Classic MPC
│   └── nengo/
│       ├── __init__.py
│       ├── base.py                 # Nengo controller base
│       ├── pid.py                  # PID + PES (not GUI compatible)
│       └── mpc.py                  # MPC + PES (not GUI compatible)
```

---

## 10. How to Run

```bash
cd soft_robotics/trajectory_tracking
./run_nengo_tracking.sh
```

Then open browser to http://localhost:8080

---

## 11. Open Question

The strain signals (proprioceptive feedback) are currently visualized but **not connected to the controller**. 

The NengoPID and NengoMPC classes implement PES (Prescribed Error Sensitivity) learning using strain feedback, but cannot run inside Nengo GUI due to simulator conflicts.

**Question**: How can PES learning be integrated to utilize the strain feedback for adaptive control, given the Nengo GUI constraint?

---

*Document generated: December 2024*
