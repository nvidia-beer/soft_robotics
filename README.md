# Soft Robotics Simulation Framework

GPU-accelerated soft robot simulation with Spiking Neural Network (SNN) control using [NVIDIA Warp](https://github.com/NVIDIA/warp) and [Nengo](https://www.nengo.ai/).

## Overview

This framework combines:
- **FEM Physics**: Implicit integration with Neo-Hookean hyperelastic materials
- **SNN Control**: Central Pattern Generators (CPG) with PES learning via Nengo
- **SDF Terrain**: Procedural signed distance field environments
- **GPU Acceleration**: NVIDIA Warp kernels for real-time simulation

## Modules

| Module | Description |
|--------|-------------|
| `rl_control/` | Locomotion demos with SDF terrain (slant, tunnel, boulder) |
| `rl_locomotion/` | CPG controllers - rate-coded and spiking (Nengo) |
| `trajectory_tracking/` | MPC + SNN adaptation for trajectory following |
| `warp/` | GPU physics solvers (implicit FEM, semi-implicit) |
| `world_map/` | SDF terrain generation and collision |
| `pygame_renderer/` | Real-time visualization |
| `tessellation/` | Delaunay mesh generation |
| `openai-gym/` | OpenAI Gym environment wrapper |

## Quick Start

### Build Docker Image
```bash
./build.sh
```

### Run Locomotion Demos
```bash
cd rl_control
./run.sh          # Interactive menu

# Or run directly
python demo_slant.py --angle 45
python demo_tunnel.py --tunnel-ratio 0.8
python demo_boulder.py --boulder-ratio 0.5
```

### Run Trajectory Tracking
```bash
cd trajectory_tracking
./run.sh

# Compare MPC vs MPC+SNN
python run_tracking.py --compare
```

### Run CPG with Nengo GUI
```bash
cd rl_locomotion
./run_nengo.sh    # Opens Nengo GUI at http://localhost:8080
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spring-Mass FEM Grid                         │
│                                                                 │
│   ●──●──●──●     Particles connected by springs                │
│   │╲/│╲/│╲/│     FEM triangles for hyperelastic response       │
│   ●──●──●──●     GPU-accelerated implicit integration          │
│   │╲/│╲/│╲/│                                                    │
│   ●──●──●──●                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SNN CPG Controller                           │
│                                                                 │
│   Hopf oscillators with phase coupling                         │
│   PES learning from strain feedback                            │
│   Traveling wave locomotion patterns                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SDF Terrain Collision                        │
│                                                                 │
│   Procedural: slants, tunnels, boulders                        │
│   Ratchet friction for directional locomotion                  │
│   GPU kernel collision detection                               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Locomotion Control
- **Hopf CPG**: Coupled oscillators generate traveling waves
- **PES Learning**: Online adaptation from strain/velocity feedback
- **Directional Friction**: Ratchet-like ground contact for forward motion

### Physics Simulation
- **Implicit FEM**: Stable large-timestep integration
- **Neo-Hookean Material**: Nonlinear hyperelastic response
- **GPU Kernels**: Warp-accelerated force computation

### Terrain Environments
- **Plane**: Flat ground locomotion baseline
- **Slant**: Inclined plane climbing (configurable angle)
- **Tunnel**: Squeeze through narrow passages
- **Boulder**: Climb over semicircular obstacles

## Requirements

- Docker with NVIDIA GPU support
- CUDA-capable GPU
- X11 display (for visualization)

## References

- [Continuous adaptive nonlinear MPC using SNNs](https://doi.org/10.1088/2634-4386/ad4209) - Halaly & Tsur 2024
- [Nengo: Large-scale brain modelling](https://www.nengo.ai/)
- [NVIDIA Warp](https://github.com/NVIDIA/warp)

## License

MIT License
