# RL Control: Soft Robot Locomotion Framework

A modular, extensible framework for soft robot locomotion testing with SDF (Signed Distance Field) terrain.

## Architecture

```
rl_control/                    # Locomotion demos
├── demo_base.py               # DemoBase class - extend for new demos
├── demo_plane.py              # Flat ground demo
├── demo_slant.py              # Slant climbing demo (with flat start)
├── demo_angled_plane.py       # Uniformly tilted plane (no flat start)
├── demo_tunnel.py             # Tunnel squeeze demo
├── demo_boulder.py            # Boulder climbing demo
└── run.sh                     # Interactive launcher

world_map/                     # SDF terrain (general module)
├── world_map.py               # WorldMap class
├── terrain_generators.py      # Procedural terrain generators
└── __init__.py                # Exports WorldMap + generators

warp/world/kernels_sdf.py      # GPU collision kernels
warp/solvers/                  # Physics solvers (SolverImplicit)
pygame_renderer/               # SDF visualization (rendering only)
```

**Key Design**: 
- Physics uses `SolverImplicit` from `warp/solvers/`
- Terrain uses `WorldMap` and generators from `world_map/`
- Collision uses kernels from `warp/world/kernels_sdf.py`

## Creating New Demos

Extend the `DemoBase` class:

```python
from rl_control import DemoBase, DemoConfig
from world_map import WorldMap, create_slant_terrain

class MyNewDemo(DemoBase):
    def __init__(self, my_param: float = 1.0, config: DemoConfig = None):
        super().__init__(config)
        self.my_param = my_param
    
    def create_terrain(self) -> WorldMap:
        # Use existing generator or create custom terrain
        return create_slant_terrain(angle_degrees=self.my_param)
    
    def get_demo_name(self) -> str:
        return f"My Demo (param={self.my_param})"

if __name__ == "__main__":
    demo = MyNewDemo(my_param=30.0)
    demo.run()
```

## Available Demos

### 1. Plane Demo (`demo_plane.py`) - Classic Locomotion
Basic flat ground locomotion - robot moves right (or configurable direction).
This is the simplest test case, equivalent to `rl_locomotion/demo_simple_cpg.py`.

```bash
python -m demo_plane                    # Default: move right
python -m demo_plane --frequency 5.0
python -m demo_plane --direction -1 0   # Move left
```

### 2. Slant Demo (`demo_slant.py`)
Climb an inclined plane against gravity.

```bash
python -m demo_slant --angle 45
python -m demo_slant --angle 30 --frequency 5.0
```

### 3. Tunnel Demo (`demo_tunnel.py`)
Squeeze through a narrow passage.

```bash
python -m demo_tunnel --tunnel-ratio 0.9
python -m demo_tunnel --tunnel-ratio 0.75 --tunnel-length 5.0
```

### 4. Boulder Demo (`demo_boulder.py`)
Climb over a semicircular obstacle.

```bash
python -m demo_boulder --boulder-ratio 0.5
python -m demo_boulder --boulder-ratio 0.7 --frequency 3.0
```

### 5. Angled Plane Demo (`demo_angled_plane.py`)
Locomotion on a uniformly tilted plane at 20° (default). Unlike Slant demo,
this has NO flat starting section - the entire surface is tilted from x=0.

```bash
python demo_angled_plane.py                 # Default: 20° angle
python demo_angled_plane.py --angle 30      # 30° tilt
python demo_angled_plane.py --angle 15 --frequency 3.0
```

## Quick Start

```bash
# Interactive menu
./run.sh

# Or run directly with Docker
docker run -it --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/..:/workspace \
    -w /workspace/rl_control \
    spring-mass-nengo \
    python -m demo_slant --angle 45
```

## Common Parameters

All demos support these via `DemoBase.add_common_args()`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--grid-size, -n` | Robot grid NxN | 4 |
| `--frequency` | CPG frequency in Hz | 4.0 |
| `--amplitude` | CPG amplitude (0-1) | 1.0 |
| `--force-scale` | Force multiplier | 20.0 |
| `--duration, -t` | Simulation time (s) | 60 |
| `--device` | cuda or cpu | cuda |

## Controls

- **Q / ESC**: Quit
- **R**: Reset simulation
- **SPACE**: Pause/Resume

## Technical Details

- **Physics**: `SolverImplicit` from `warp/solvers/`
- **Terrain**: `WorldMap` from `world_map/`
- **Collision**: `apply_sdf_boundary_with_friction_2d` from `warp/world/`
- **Locomotion**: Classic CPG (Hopf oscillator) - no SNN
- **Rendering**: `pygame_renderer.Renderer` for SDF visualization
