#!/usr/bin/env python3
"""
Figure 1: Analysis of Neuromorphic PID Control

Compares P, PI, and PD control using SNN-based implementation.
Demonstrates the response characteristics of each control variant.

The simulation initializes with volume ratio = 1.0 (rest state)
and inflates to target volume ratio = 2.0 (inflated state).

Output:
    figures/figure_snn.png
    figures/figure_snn.npz (raw data)

Usage:
    python figure_snn.py                    # Run with default parameters
    python figure_snn.py --total-time 15    # Custom simulation time
    python figure_snn.py --no-sim           # Plot from saved data only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import warp as wp
import nengo

from models import BalloonModel
from solvers import SolverImplicitFEM
from controllers.nengo import SNN_PID_Controller


# ============================================================================
# Configuration
# ============================================================================

# Controller configurations: (name, Kp, Ki, Kd, color)
# Controller configurations: (name, Kp, Ki, Kd, label)
# Now SNN-PID actually controls inflation (baseline=1.0)
#
# GAIN CALCULATION (based on classic PID values from figure_classic.py):
# The SNN uses the SAME Kp/Ki/Kd values as classic PID but they undergo
# NEF normalization inside the network (Zaidel et al. 2021):
#
#   Kp_norm = Kp * error_scale / output_scale
#   where error_scale=0.5 (max volume error), output_scale=10.0 (u_max)
#
# The output is then scaled back up by output_scale, so the effective
# steady-state gain is approximately equal to the input Kp/Ki/Kd.
#
# However, the SNN has additional neural dynamics:
#   - Integrator: leaky (0.95 decay) with tau_int=0.1s, gain tau_int*0.5
#   - Derivative: fast/slow synapse difference (tau_fast=0.005, tau_slow=0.2)
#   - Synaptic delays: tau_syn=0.01s on all connections
#
# These dynamics mean the SNN response will differ slightly from classic PID:
#   - Slower response due to synaptic time constants
#   - Smoother output due to neural filtering
#   - Some spiking noise in the output
#
# FAIR COMPARISON: Uses SAME gains as classic PID (loaded from pid_gains.txt)
# Only difference is which terms are enabled (0 = disabled).
#

def load_snn_gains():
    """Load SNN PID gains from snn_gains.txt.
    
    SNN needs MUCH lower gains than classic PID due to neural dynamics.
    Run ./run_snn_tuning.sh to find optimal values.
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snn_gains.txt')
    
    # Default values (conservative)
    gains = {'KP': 2.0, 'KI': 0.6, 'KD': 0.2}
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                if key in gains:
                    gains[key] = float(value.strip())
        print(f"  Loaded SNN gains from: {config_file}")
    else:
        print(f"  Using default SNN gains (run ./run_snn_tuning.sh to tune)")
    
    kp, ki, kd = gains['KP'], gains['KI'], gains['KD']
    print(f"  SNN gains: Kp={kp}, Ki={ki}, Kd={kd}")
    
    return kp, ki, kd

KP, KI, KD = load_snn_gains()

# Use SAME u_max as classic PID
U_MAX_SNN = 10.0

CONTROLLER_CONFIGS = [
    ('P',   KP, 0.0, 0.0, 'P'),    # P only: has offset
    ('PI',  KP, KI,  0.0, 'PI'),   # PI: integral eliminates offset
    ('PD',  KP, 0.0, KD,  'PD'),   # PD: damped, still has offset
    ('PID', KP, KI,  KD,  'PID'),  # PID: all terms combined
]

# Time constants (from Zaidel et al. 2021)
TAU_INT = 0.01     # τi: Integrator time constant (fast)
TAU_SLOW = 0.005   # τd slow: Derivative slow synapse
TAU_FAST = 0.0005  # τd fast: Derivative fast synapse
TAU_SYN = 0.001    # τs: Synaptic time constant (minimal)

# Figure output directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def run_simulation(
    controller_name: str,
    Kp: float,
    Ki: float,
    Kd: float,
    total_time: float = 5.0,  # 500 steps
    dt: float = 0.01,
    nengo_dt: float = 0.001,  # 1ms for smooth SNN operation
    initial_volume_ratio: float = 2.0,
    target_volume_ratio: float = 1.0,
    n_neurons: int = 2000,  # More neurons = less noise
    device: str = 'cuda',
    seed: int = 42,
):
    """
    Run a single simulation with specified controller configuration.
    
    Args:
        controller_name: Name for logging
        Kp, Ki, Kd: PID gains
        total_time: Simulation duration (seconds)
        dt: Physics time step
        nengo_dt: Nengo time step
        initial_volume_ratio: Starting volume (relative to rest)
        target_volume_ratio: Target volume (relative to rest)
        n_neurons: Neurons per ensemble
        device: 'cuda' or 'cpu'
        seed: Random seed
    
    Returns:
        dict with time series data
    """
    np.random.seed(seed)
    
    # Initialize Warp
    wp.init()
    
    print(f"\n{'='*60}")
    print(f"Running: {controller_name} Controller (Kp={Kp}, Ki={Ki}, Kd={Kd})")
    print(f"{'='*60}")
    
    # Create model
    model = BalloonModel(
        radius=0.5,
        num_boundary=16,
        num_rings=2,
        max_volume_ratio=max(initial_volume_ratio, target_volume_ratio) + 0.5,
        device=device,
        boxsize=3.0,
        spring_stiffness=1000.0,
        spring_damping=5.0,
        fem_E=2000.0,
        fem_nu=0.45,
        fem_damping=10.0,
    )
    
    # Create solver
    solver = SolverImplicitFEM(
        model,
        dt=dt,
        mass=1.0,
        preconditioner_type="diag",
        solver_type="bicgstab",
        max_iterations=50,
        tolerance=1e-4,
        rebuild_matrix_every=1,
    )
    
    # Create states
    state = model.state()
    state_next = model.state()
    
    # Pre-inflate to initial volume
    print(f"  Pre-inflating to volume ratio = {initial_volume_ratio}...")
    for _ in range(100):
        current_vol = model.compute_current_volume(state)
        target_vol = model.initial_volume * initial_volume_ratio
        if abs(current_vol - target_vol) < 0.01:
            break
        
        # Simple proportional inflation
        error = target_vol - current_vol
        pressure = np.clip(error * 5.0, -10, 10)
        model.apply_inflation(state, pressure, initial_volume_ratio)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    actual_initial_ratio = model.compute_current_volume(state) / model.initial_volume
    print(f"  Initial volume ratio achieved: {actual_initial_ratio:.3f}")
    
    # =========================================================================
    # Build Nengo Network
    # =========================================================================
    
    current_error = [0.0]  # This will store NORMALIZED error (volume ratio error)
    snn_output = [0.0]
    last_pressure = [0.0]  # For rate limiting
    MAX_PRESSURE_RATE = 1.0  # Rate limit (1.0 works well, tighter breaks I and D terms)
    # Error scale: 1.0 to match ratio error range (0 to ~1.0)
    error_scale = 1.0
    
    def get_error(t):
        # Error is already normalized to volume ratio, so just scale to [-1, 1]
        return current_error[0] / error_scale
    
    nengo_model = nengo.Network(label=f"SNN_{controller_name}")
    
    with nengo_model:
        controller = SNN_PID_Controller(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            u_max=U_MAX_SNN,  # Scaled to keep Kp_norm <= 1.0
            n_neurons=n_neurons,
            error_scale=error_scale,  # 1.0 (matches ratio error range)
        )
        controller.build(
            model=nengo_model,
            get_error_callback=get_error,
        )
        
        def sync_output(t):
            snn_output[0] = controller.get_output()
        
        nengo.Node(sync_output, size_in=0, size_out=0, label="Output_Sync")
    
    sim = nengo.Simulator(nengo_model, dt=nengo_dt)
    
    # =========================================================================
    # Run Control Simulation
    # =========================================================================
    
    n_steps = int(total_time / dt)
    physics_steps_per_nengo = int(dt / nengo_dt)
    
    # Data storage
    times = []
    volumes = []
    errors = []
    pressures = []
    
    print(f"  Running {n_steps} steps (target ratio = {target_volume_ratio})...")
    
    for step in range(n_steps):
        # Current state
        current_volume = model.compute_current_volume(state)
        current_ratio = current_volume / model.initial_volume
        
        # Compute error in volume RATIO space (stays bounded in [-1, 1] for reasonable targets)
        # This ensures the SNN input stays within ensemble radius
        error_ratio = target_volume_ratio - current_ratio
        current_error[0] = error_ratio  # Store ratio error for SNN
        
        # Run Nengo steps
        for _ in range(physics_steps_per_nengo):
            sim.step()
        
        # Get pressure with rate limiting (reduces spikiness without delay)
        raw_pressure = snn_output[0]
        delta = raw_pressure - last_pressure[0]
        delta = np.clip(delta, -MAX_PRESSURE_RATE, MAX_PRESSURE_RATE)
        pressure = last_pressure[0] + delta
        last_pressure[0] = pressure
        
        # Apply inflation control - use 1.0 as baseline so SNN controls everything
        model.apply_inflation(state, pressure, 1.0)
        
        # Physics step
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
        
        # Store data (normalized)
        sim_time = step * dt
        times.append(sim_time)
        volumes.append(current_ratio)  # Volume ratio
        errors.append(error_ratio)  # Volume ratio error (already normalized)
        pressures.append(pressure)
        
        # Progress
        if step % 200 == 0:
            print(f"    Step {step:4d}: vol={current_ratio:.3f} err={error_ratio:.4f} p={pressure:.2f}")
    
    sim.close()
    
    # Final state
    final_vol = model.compute_current_volume(state) / model.initial_volume
    print(f"  Final volume ratio: {final_vol:.3f} (target: {target_volume_ratio:.3f})")
    
    return {
        'name': controller_name,
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd,
        'times': np.array(times),
        'volumes': np.array(volumes),
        'errors': np.array(errors),
        'pressures': np.array(pressures),
        'initial_ratio': initial_volume_ratio,
        'target_ratio': target_volume_ratio,
    }


def run_all_simulations(args):
    """Run simulations for all controller configurations."""
    
    results = {}
    
    for name, Kp, Ki, Kd, _ in CONTROLLER_CONFIGS:
        data = run_simulation(
            controller_name=name,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            total_time=args.total_time,
            dt=args.dt,
            nengo_dt=args.nengo_dt,
            initial_volume_ratio=args.initial_volume,
            target_volume_ratio=args.target_volume,
            n_neurons=args.neurons,
            device=args.device,
            seed=args.seed,
        )
        results[name] = data
    
    # Save data
    os.makedirs(FIGURES_DIR, exist_ok=True)
    data_file = os.path.join(FIGURES_DIR, 'figure_snn.npz')
    
    np.savez(
        data_file,
        **{f"{k}_times": v['times'] for k, v in results.items()},
        **{f"{k}_volumes": v['volumes'] for k, v in results.items()},
        **{f"{k}_errors": v['errors'] for k, v in results.items()},
        **{f"{k}_pressures": v['pressures'] for k, v in results.items()},
        controller_names=[c[0] for c in CONTROLLER_CONFIGS],
        controller_params=[(c[1], c[2], c[3]) for c in CONTROLLER_CONFIGS],
        initial_volume=args.initial_volume,
        target_volume=args.target_volume,
        tau_int=TAU_INT,
        tau_slow=TAU_SLOW,
        tau_fast=TAU_FAST,
    )
    print(f"\nData saved: {data_file}")
    
    return results


def create_figure(results, args):
    """
    Create 4-panel comparison figure (P, PI, PD, PID).
    
    Layout: 1 row x 4 columns
    Each subplot shows Volume, Error, and Pressure - ACTUAL VALUES (same as classic PID).
    """
    
    # Figure style settings
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
    })
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.12, 
                        wspace=0.08)
    
    # Plot each controller in its own subplot
    for col_idx, (name, Kp, Ki, Kd, _) in enumerate(CONTROLLER_CONFIGS):
        ax = axes[col_idx]
        data = results[name]
        times = data['times']
        volumes = data['volumes']  # Volume ratio (1.0 to 2.0)
        errors = data['errors']    # Volume ratio error
        pressures = data['pressures']  # Pressure (-10 to 10)
        
        # Plot YELLOW/ORANGE PRESSURE FIRST (so it's behind)
        ax.plot(times, pressures, color='#FFAA00', linewidth=2, 
                label='Pressure', zorder=1)
        
        # Plot RED ERROR
        ax.plot(times, errors, color='#CC0000', linewidth=1.5, 
                label='Error', zorder=2)
        
        # Plot BLACK VOLUME (PID response)
        ax.plot(times, volumes, color='black', linewidth=2, 
                label=name, zorder=3)
        
        # Target line - RED DASHED at y=2.0 (actual target)
        ax.axhline(y=2.0, color='#CC0000', linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        # Calculate final error
        final_vol = volumes[-1]
        final_error = abs(2.0 - final_vol)
        
        ax.set_title(f'{name} Control\n(error: {final_error:.3f})', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col_idx == 0 else '')
        ax.grid(True, alpha=0.3)
        
        # Legend in lower right
        ax.legend(loc='lower right', fontsize=8)
    
    # Global title with parameters
    params_str = f"Kp={KP}, Ki={KI}, Kd={KD}"
    
    fig.suptitle(
        f"Analysis of Neuromorphic PID Control (SNN)\n"
        f"{params_str}",
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # Save figure
    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_file = os.path.join(FIGURES_DIR, 'figure_snn.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {output_file}")
    
    # Also save as PDF for publication
    pdf_file = os.path.join(FIGURES_DIR, 'figure_snn.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"PDF saved: {pdf_file}")
    
    plt.close()
    
    return output_file


def plot_from_data(data_file, args):
    """Load saved data and create figure."""
    
    print(f"Loading data from: {data_file}")
    data = np.load(data_file, allow_pickle=True)
    
    # Reconstruct results dict
    controller_names = data['controller_names']
    results = {}
    
    for name in controller_names:
        results[name] = {
            'name': name,
            'times': data[f'{name}_times'],
            'volumes': data[f'{name}_volumes'],
            'errors': data[f'{name}_errors'],
            'pressures': data[f'{name}_pressures'],
        }
    
    return create_figure(results, args)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 1: Neuromorphic PID Control Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python figure_snn.py                      # Run full simulation and plot
  python figure_snn.py --total-time 15      # Longer simulation
  python figure_snn.py --no-sim             # Plot from saved data
  python figure_snn.py --device cpu         # Use CPU instead of GPU
        """
    )
    
    # Simulation parameters
    parser.add_argument('--total-time', '-t', type=float, default=5.0,
                       help='Total simulation time in seconds (default: 5)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Physics time step (default: 0.01)')
    parser.add_argument('--nengo-dt', type=float, default=0.001,
                       help='Nengo time step (default: 0.001)')
    
    # Initial conditions
    parser.add_argument('--initial-volume', type=float, default=1.0,
                       help='Initial volume ratio (default: 1.0)')
    parser.add_argument('--target-volume', type=float, default=2.0,
                       help='Target volume ratio (default: 2.0)')
    
    # Neural network
    parser.add_argument('--neurons', type=int, default=2000,
                       help='Neurons per ensemble (default: 2000)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Compute device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Run options
    parser.add_argument('--no-sim', action='store_true',
                       help='Skip simulation, only plot from saved data')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Data file to plot (default: figure_snn.npz)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  FIGURE 1: Analysis of Neuromorphic PID Control")
    print("=" * 70)
    print(f"\nController configurations:")
    for name, Kp, Ki, Kd, _ in CONTROLLER_CONFIGS:
        print(f"  {name:3s}: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"\nTime constants:")
    print(f"  τi (integral): {TAU_INT}s")
    print(f"  τd (derivative): {TAU_FAST}s - {TAU_SLOW}s")
    
    if args.no_sim:
        # Plot from saved data
        data_file = args.data_file or os.path.join(FIGURES_DIR, 'figure_snn.npz')
        if not os.path.exists(data_file):
            print(f"\nError: Data file not found: {data_file}")
            print("Run without --no-sim first to generate data.")
            return 1
        
        plot_from_data(data_file, args)
    else:
        # Run simulations and create figure
        print(f"\nSimulation settings:")
        print(f"  Initial volume ratio: {args.initial_volume}")
        print(f"  Target volume ratio: {args.target_volume}")
        print(f"  Total time: {args.total_time}s")
        print(f"  Device: {args.device}")
        
        results = run_all_simulations(args)
        create_figure(results, args)
    
    print("\n" + "=" * 70)
    print("  Figure generation complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
