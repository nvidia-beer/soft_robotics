#!/usr/bin/env python3
"""
Figure PES: Analysis of PES (Prescriptive Error-Driven Synapses) Learning Rates

Compares different PES learning rates on SNN PID controller.
PES learns strain → pressure feedforward mapping to improve control.

Based on: Zaidel et al., "Neuromorphic NEF-Based Inverse Kinematics and PID Control"
Frontiers in Neurorobotics, 2021

Output:
    figures/figure_pes.png
    figures/figure_pes.pdf
    figures/figure_pes.npz (raw data)

Usage:
    python figure_pes.py                    # Run with default parameters
    python figure_pes.py --no-sim           # Plot from saved data only
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
from controllers.stress import SNN_Stress_Controller


# ============================================================================
# Configuration
# ============================================================================

def load_snn_gains():
    """Load SNN PID gains from snn_gains.txt."""
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
        print(f"  Loaded SNN gains from: {config_file}")
    
    return gains['KP'], gains['KI'], gains['KD']

KP, KI, KD = load_snn_gains()
U_MAX = 10.0

# Controller configurations: (name, Kp, Kd, pes_lr, label)
# SNN PID with different PES learning rates
CONTROLLER_CONFIGS = [
    ('PES_0',    KP, KD, 0.0,   'No PES (PD only)'),
    ('PES_1e-5', KP, KD, 1e-5,  'PES lr=1e-5'),
    ('PES_1e-4', KP, KD, 1e-4,  'PES lr=1e-4'),
    ('PES_1e-3', KP, KD, 1e-3,  'PES lr=1e-3'),
]

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def run_simulation(
    controller_name: str,
    Kp: float,
    Kd: float,
    pes_lr: float,
    total_time: float = 5.0,  # 500 steps
    dt: float = 0.01,
    nengo_dt: float = 0.001,
    initial_volume_ratio: float = 1.0,
    target_volume_ratio: float = 2.0,
    n_neurons: int = 2000,
    device: str = 'cuda',
    seed: int = 42,
):
    """Run simulation with stress controller."""
    
    wp.init()
    np.random.seed(seed)
    
    print(f"\n  Running {controller_name} (Kp={Kp}, Kd={Kd}, PES_lr={pes_lr})...")
    
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
    
    # Build SNN Stress Controller
    current_error = [0.0]
    snn_output = [0.0]
    current_strain = [np.zeros(7)]  # 7D strain (5 springs + 2 FEMs)
    last_pressure = [0.0]
    MAX_PRESSURE_RATE = 1.0
    error_scale = 1.0
    
    def get_error(t):
        return current_error[0] / error_scale
    
    def get_strain(t):
        return current_strain[0]
    
    nengo_model = nengo.Network(label=f"Stress_{controller_name}")
    with nengo_model:
        controller = SNN_Stress_Controller(
            Kp=Kp,
            Kd=Kd,
            u_max=U_MAX,
            n_neurons=n_neurons,
            error_scale=error_scale,
            strain_dim=7,
            pes_learning_rate=pes_lr,
        )
        controller.build(
            model=nengo_model,
            get_error_callback=get_error,
            get_strain_callback=get_strain if pes_lr > 0 else None,
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
        
        # Get strain from model (simplified - use volume ratio as proxy)
        # In a real implementation, this would come from FEM stress tensors
        strain_proxy = np.clip((current_ratio - 1.5) * 2.0, -1, 1)
        current_strain[0] = np.full(7, strain_proxy)
        
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
        'name': controller_name,
        'Kp': Kp,
        'Kd': Kd,
        'pes_lr': pes_lr,
        'times': np.array(times),
        'volumes': np.array(volumes),
        'errors': np.array(errors),
        'pressures': np.array(pressures),
    }


def run_all_simulations(args):
    """Run simulations for all controller configurations."""
    
    results = {}
    
    for name, Kp, Kd, pes_lr, _ in CONTROLLER_CONFIGS:
        data = run_simulation(
            controller_name=name,
            Kp=Kp,
            Kd=Kd,
            pes_lr=pes_lr,
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
    data_file = os.path.join(FIGURES_DIR, 'figure_pes.npz')
    
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
        n_neurons=args.neurons,
    )
    print(f"\nData saved: {data_file}")
    
    return results


def create_figure(results, args):
    """Create 4-panel comparison figure."""
    
    # Figure style
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
    
    for col_idx, (name, Kp, Kd, pes_lr, label) in enumerate(CONTROLLER_CONFIGS):
        ax = axes[col_idx]
        data = results[name]
        times = data['times']
        volumes = data['volumes']
        errors = data['errors']
        pressures = data['pressures']
        
        # Plot
        ax.plot(times, pressures, color='#FFAA00', linewidth=2, 
                label='Pressure', zorder=1)
        ax.plot(times, errors, color='#CC0000', linewidth=1.5, 
                label='Error', zorder=2)
        ax.plot(times, volumes, color='black', linewidth=2, 
                label=label, zorder=3)
        ax.axhline(y=2.0, color='#CC0000', linestyle='--', linewidth=1.5, 
                   label='Target', zorder=4)
        
        final_error = abs(2.0 - volumes[-1])
        ax.set_title(f'{label}\n(error: {final_error:.3f})', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, 3)
        ax.set_ylabel('Value' if col_idx == 0 else '')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    
    params_str = f"Kp={KP}, Kd={KD}, n_neurons={args.neurons}"
    fig.suptitle(
        f"Analysis of Neuromorphic Stress Controller (SNN + PES)\n{params_str}",
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # Save
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_file = os.path.join(FIGURES_DIR, 'figure_pes.png')
    pdf_file = os.path.join(FIGURES_DIR, 'figure_pes.pdf')
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nFigure saved: {png_file}")
    print(f"PDF saved: {pdf_file}")
    
    return png_file


def plot_from_data(data_file, args):
    """Load saved data and create figure."""
    
    print(f"Loading data from: {data_file}")
    data = np.load(data_file, allow_pickle=True)
    
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
    
    # Update args with saved params
    args.neurons = int(data.get('n_neurons', 2000))
    
    return create_figure(results, args)


def main():
    parser = argparse.ArgumentParser(description='Figure 5: Stress Controller Analysis')
    
    parser.add_argument('--total-time', type=float, default=5.0,
                       help='Total simulation time (default: 5.0 = 500 steps)')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--nengo-dt', type=float, default=0.001)
    parser.add_argument('--initial-volume', type=float, default=1.0)
    parser.add_argument('--target-volume', type=float, default=2.0)
    parser.add_argument('--neurons', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-sim', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Figure 5: Neuromorphic Stress Controller (SNN + PES)")
    print("=" * 60)
    print(f"\n  Kp={KP}, Kd={KD}")
    print(f"  Neurons: {args.neurons} per ensemble")
    
    if args.no_sim:
        data_file = os.path.join(FIGURES_DIR, 'figure_pes.npz')
        if os.path.exists(data_file):
            plot_from_data(data_file, args)
        else:
            print(f"Data file not found: {data_file}")
            print("Run without --no-sim first.")
    else:
        results = run_all_simulations(args)
        create_figure(results, args)
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
