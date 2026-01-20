#!/usr/bin/env python3
"""
Figure 3: Effect of Neuron Count on SNN PID Control

Compares PID control performance with different numbers of neurons.
Shows how neural noise decreases with more neurons.

Output:
    figures/figure_neuron_count.png
    figures/figure_neuron_count.pdf
    figures/figure_neuron_count.npz (raw data)

Usage:
    python figure_neuron_count.py                    # Run with default parameters
    python figure_neuron_count.py --no-sim           # Plot from saved data only
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

import warp as wp
import nengo

from models import BalloonModel
from solvers import SolverImplicitFEM
from controllers.nengo import SNN_PID_Controller


# ============================================================================
# Configuration
# ============================================================================

# Neuron counts to compare
NEURON_COUNTS = [250, 500, 1000, 2000]

# Load gains from snn_gains.txt
def load_snn_gains():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snn_gains.txt')
    gains = {'KP': 4.5, 'KI': 1.25, 'KD': 0.05}
    
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
    
    return gains['KP'], gains['KI'], gains['KD']

KP, KI, KD = load_snn_gains()
U_MAX = 10.0

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def run_simulation(
    n_neurons: int,
    total_time: float = 5.0,  # 500 steps
    dt: float = 0.01,
    nengo_dt: float = 0.001,
    initial_volume_ratio: float = 1.0,
    target_volume_ratio: float = 2.0,
    device: str = 'cuda',
    seed: int = 42,
):
    """Run simulation with specified neuron count."""
    
    wp.init()
    np.random.seed(seed)
    
    print(f"\n  Running PID with {n_neurons} neurons...")
    
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
    
    state = model.state()
    state_next = model.state()
    
    # Pre-inflate to initial ratio
    target_vol = model.initial_volume * initial_volume_ratio
    for _ in range(100):
        current_vol = model.compute_current_volume(state)
        if abs(current_vol - target_vol) < 0.01:
            break
        error = target_vol - current_vol
        pressure = np.clip(error * 5.0, -10, 10)
        model.apply_inflation(state, pressure, initial_volume_ratio)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    # Build SNN controller
    current_error = [0.0]
    snn_output = [0.0]
    last_pressure = [0.0]
    MAX_PRESSURE_RATE = 1.0
    error_scale = 1.0
    
    def get_error(t):
        return current_error[0] / error_scale
    
    nengo_model = nengo.Network(label=f"SNN_PID_{n_neurons}")
    with nengo_model:
        controller = SNN_PID_Controller(
            Kp=KP,
            Ki=KI,
            Kd=KD,
            u_max=U_MAX,
            n_neurons=n_neurons,
            error_scale=error_scale,
        )
        controller.build(
            model=nengo_model,
            get_error_callback=get_error,
        )
        
        def sync_output(t):
            snn_output[0] = controller.get_output()
        
        nengo.Node(sync_output, size_in=0, size_out=0, label="Output_Sync")
    
    sim = nengo.Simulator(nengo_model, dt=nengo_dt)
    
    # Run simulation
    n_steps = int(total_time / dt)
    physics_steps_per_nengo = max(1, int(dt / nengo_dt))
    
    times = []
    volumes = []
    errors = []
    pressures = []
    
    for step in range(n_steps):
        current_volume = model.compute_current_volume(state)
        current_ratio = current_volume / model.initial_volume
        
        error_ratio = target_volume_ratio - current_ratio
        current_error[0] = error_ratio
        
        for _ in range(physics_steps_per_nengo):
            sim.step()
        
        # Rate limiting
        raw_pressure = snn_output[0]
        delta = raw_pressure - last_pressure[0]
        delta = np.clip(delta, -MAX_PRESSURE_RATE, MAX_PRESSURE_RATE)
        pressure = last_pressure[0] + delta
        last_pressure[0] = pressure
        
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
        
        sim_time = step * dt
        times.append(sim_time)
        volumes.append(current_ratio)
        errors.append(error_ratio)
        pressures.append(pressure)
        
        if step % 200 == 0:
            print(f"    Step {step:4d}: vol={current_ratio:.3f} err={error_ratio:.4f} p={pressure:.2f}")
    
    sim.close()
    
    final_vol = model.compute_current_volume(state) / model.initial_volume
    print(f"  Final volume ratio: {final_vol:.3f} (target: {target_volume_ratio:.3f})")
    
    return {
        'n_neurons': n_neurons,
        'times': np.array(times),
        'volumes': np.array(volumes),
        'errors': np.array(errors),
        'pressures': np.array(pressures),
    }


def run_all_simulations(args):
    """Run simulations for all neuron counts."""
    
    results = {}
    
    for n_neurons in NEURON_COUNTS:
        data = run_simulation(
            n_neurons=n_neurons,
            total_time=args.total_time,
            dt=args.dt,
            nengo_dt=args.nengo_dt,
            initial_volume_ratio=args.initial_volume,
            target_volume_ratio=args.target_volume,
            device=args.device,
            seed=args.seed,
        )
        results[n_neurons] = data
    
    # Save data
    os.makedirs(FIGURES_DIR, exist_ok=True)
    data_file = os.path.join(FIGURES_DIR, 'figure_neuron_count.npz')
    
    np.savez(
        data_file,
        neuron_counts=NEURON_COUNTS,
        **{f"n{n}_times": results[n]['times'] for n in NEURON_COUNTS},
        **{f"n{n}_volumes": results[n]['volumes'] for n in NEURON_COUNTS},
        **{f"n{n}_errors": results[n]['errors'] for n in NEURON_COUNTS},
        **{f"n{n}_pressures": results[n]['pressures'] for n in NEURON_COUNTS},
        gains=[KP, KI, KD],
        initial_volume=args.initial_volume,
        target_volume=args.target_volume,
    )
    print(f"\nData saved: {data_file}")
    
    return results


def create_figure(data_file: str = None):
    """Create the comparison figure - same style as figure_snn and figure_classic."""
    
    if data_file is None:
        data_file = os.path.join(FIGURES_DIR, 'figure_neuron_count.npz')
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Run without --no-sim first to generate data.")
        return
    
    data = np.load(data_file, allow_pickle=True)
    neuron_counts = list(data['neuron_counts'])
    gains = data['gains']
    
    print(f"\nCreating figure from: {data_file}")
    print(f"Neuron counts: {neuron_counts}")
    print(f"Gains: Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]}")
    
    # Figure style settings (same as classic PID)
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
    })
    
    # Create 1x4 figure (same layout as classic PID)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.12, 
                        wspace=0.08)
    
    for col_idx, n_neurons in enumerate(neuron_counts):
        ax = axes[col_idx]
        
        times = data[f'n{n_neurons}_times']
        volumes = data[f'n{n_neurons}_volumes']
        pressures = data[f'n{n_neurons}_pressures']
        errors = data[f'n{n_neurons}_errors']
        
        # Plot YELLOW/ORANGE PRESSURE FIRST (so it's behind)
        ax.plot(times, pressures, color='#FFAA00', linewidth=2, 
                label='Pressure', zorder=1)
        
        # Plot RED ERROR
        ax.plot(times, errors, color='#CC0000', linewidth=1.5, 
                label='Error', zorder=2)
        
        # Plot BLACK VOLUME (PID response)
        ax.plot(times, volumes, color='black', linewidth=2, 
                label='PID', zorder=3)
        
        # Target line - RED DASHED at y=2.0 (actual target)
        ax.axhline(y=2.0, color='#CC0000', linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        # Calculate final error
        final_vol = volumes[-1]
        final_error = abs(2.0 - final_vol)
        
        ax.set_title(f'PID Control ({n_neurons} neurons)\n(error: {final_error:.3f})', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col_idx == 0 else '')
        ax.grid(True, alpha=0.3)
        
        # Legend in lower right
        ax.legend(loc='lower right', fontsize=8)
    
    # Global title with parameters
    params_str = f"Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]}"
    
    fig.suptitle(
        f"Effect of Neuron Count on SNN PID Control\n"
        f"{params_str}",
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # Save
    png_file = os.path.join(FIGURES_DIR, 'figure_neuron_count.png')
    pdf_file = os.path.join(FIGURES_DIR, 'figure_neuron_count.pdf')
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nFigure saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Summary: Effect of Neuron Count on SNN PID")
    print("=" * 60)
    print(f"{'Neurons':>8} | {'Final':>8} | {'Osc':>8} | {'P_std':>8}")
    print("-" * 45)
    
    for n_neurons in neuron_counts:
        volumes = data[f'n{n_neurons}_volumes']
        pressures = data[f'n{n_neurons}_pressures']
        osc = np.std(volumes[-100:])
        final = volumes[-1]
        p_std = np.std(pressures)
        print(f'{n_neurons:>8} | {final:>8.3f} | {osc:>8.4f} | {p_std:>8.3f}')


def main():
    parser = argparse.ArgumentParser(description='Figure 3: Neuron Count Comparison')
    
    # Simulation parameters
    parser.add_argument('--total-time', type=float, default=5.0,
                       help='Total simulation time (default: 5.0 = 500 steps)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Physics time step (default: 0.01)')
    parser.add_argument('--nengo-dt', type=float, default=0.001,
                       help='Nengo time step (default: 0.001)')
    parser.add_argument('--initial-volume', type=float, default=1.0,
                       help='Initial volume ratio (default: 1.0)')
    parser.add_argument('--target-volume', type=float, default=2.0,
                       help='Target volume ratio (default: 2.0)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Options
    parser.add_argument('--no-sim', action='store_true',
                       help='Skip simulation, only create figure from saved data')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Figure 3: Effect of Neuron Count on SNN PID")
    print("=" * 60)
    print(f"\n  Neuron counts: {NEURON_COUNTS}")
    print(f"  Gains: Kp={KP}, Ki={KI}, Kd={KD}")
    
    if not args.no_sim:
        run_all_simulations(args)
    
    create_figure()
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
