# RL Control with Hopf CPG

Spiking and rate-coded Central Pattern Generator (CPG) for spring-mass locomotion.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SNN CPG with PES Learning                                                  │
│                                                                             │
│  position → vel_ens (differentiator) ───┐                                  │
│                                          │ error = strain - velocity        │
│  strain → strain_ens ═══════[PES]═══════╪═══► osc_ens → output             │
│                           (learns!)      │                                  │
│                                                                             │
│  PES learns: "given strain, how to modulate for max movement"              │
│    - High strain + moving → low error → keep going                          │
│    - High strain + stuck  → high error → adjust weights                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `cpg.py` | Rate-coded HopfCPG (simple, no spiking) |
| `snn/cpg.py` | Spiking HopfCPG with PES learning |
| `demo_snn_gui.py` | Nengo GUI demo with physics |
| `run_snn.sh` | Run script for Docker |

## Quick Start

### Run the Demo
```bash
cd soft_robotics/rl_locomotion
./run_snn.sh
# Opens Nengo GUI at http://localhost:8080
```

### Use in Code

**SNN CPG (default, with PES learning):**
```python
from rl_locomotion import CPG

cpg = CPG(num_groups=9, frequency=2.0, pes_learning_rate=1e-4)
cpg.set_position(group_x_positions)  # For velocity computation
cpg.set_feedback(strain_per_group)   # For PES learning
output = cpg.step()                   # [-1, 1] per group
```

**Rate-coded CPG (simpler, no learning):**
```python
from rl_locomotion import HopfCPG

cpg = HopfCPG(num_groups=9, frequency=2.0, direction=1.0)
output = cpg(t)  # [-1, 1] per group
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_groups` | 9 | Number of 2x2 groups (3x3 for 4x4 grid) |
| `frequency` | 2.0 | Oscillation frequency (Hz) |
| `amplitude` | 0.5 | Output amplitude [0, 1] |
| `direction` | 1.0 | +1 = body right, -1 = body left |
| `coupling_strength` | 2.0 | Inter-oscillator coupling |
| `pes_learning_rate` | 1e-4 | PES learning rate (SNN only) |
| `strain_gain` | 0.5 | Strain contribution to error (SNN only) |

## Hopf Dynamics

```
ṙ = a(μ - r²)r              # Amplitude converges to √μ
θ̇ = d·ω + coupling          # Phase with neighbor coupling
x = r·cos(θ)                # Output
```

Where `d` is direction (+1/-1) and coupling creates traveling waves.
