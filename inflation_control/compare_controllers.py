#!/usr/bin/env python3
"""
Compare SNN_PID vs SNN_Stress Controllers for Inflation Control.

Demonstrates that PES learning in SNN_Stress improves performance over time
by learning the strain→pressure mapping.

Key comparison:
- SNN_PID: Pure spiking PD control (no learning)
- SNN_Stress: Spiking PD + PES feedforward (learns strain→pressure)

Expected results:
- Early phase: Both controllers similar (PES hasn't learned yet)
- Late phase: SNN_Stress should outperform SNN_PID (PES has learned)

Usage:
    python compare_controllers.py                    # Run comparison
    python compare_controllers.py --plot-file X.npz  # Plot from saved data
    python compare_controllers.py --total-time 30    # Longer simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from datetime import datetime
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warp as wp
import nengo

from models import BalloonModel
from solvers import SolverImplicitFEM
from controllers.nengo import SNN_PID_Controller
from controllers.stress import SNN_Stress_Controller


def create_target_sequence(total_time, dt, max_volume=2.0, pattern='step'):
    """
    Create a target volume sequence.
    
    Args:
        total_time: Total simulation time (seconds)
        dt: Time step
        max_volume: Maximum volume ratio
        pattern: 'step', 'ramp', 'sine', or 'multi_step'
    
    Returns:
        Array of target volume ratios
    """
    n_steps = int(total_time / dt)
    targets = np.ones(n_steps)
    
    if pattern == 'step':
        # Single step at 20% of total time
        step_idx = int(0.2 * n_steps)
        targets[step_idx:] = max_volume
        
    elif pattern == 'ramp':
        # Linear ramp
        targets = np.linspace(1.0, max_volume, n_steps)
        
    elif pattern == 'sine':
        # Sinusoidal oscillation
        t = np.linspace(0, total_time, n_steps)
        amplitude = (max_volume - 1.0) / 2
        targets = 1.0 + amplitude + amplitude * np.sin(2 * np.pi * 0.1 * t)
        
    elif pattern == 'multi_step':
        # Multiple steps to challenge learning
        step_times = [0.1, 0.3, 0.5, 0.7]  # Fraction of total time
        step_values = [1.5, max_volume, 1.3, max_volume]
        current_target = 1.0
        for i, frac in enumerate(step_times):
            idx = int(frac * n_steps)
            current_target = step_values[i]
            targets[idx:] = current_target
    
    return targets


def run_single_simulation(
    controller_type,  # 'snn_pid' or 'snn_stress'
    target_sequence,
    dt=0.01,
    nengo_dt=0.001,
    max_volume=2.0,
    pes_learning_rate=1e-4,
    n_neurons=100,
    device='cuda',
    seed=0,
):
    """
    Run a single simulation with specified controller.
    
    Returns:
        dict with time series data
    """
    np.random.seed(seed)
    
    # Initialize Warp
    wp.init()
    
    # Create model
    model = BalloonModel(
        radius=0.5,
        num_boundary=16,
        num_rings=2,
        max_volume_ratio=max_volume,
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
    
    # Get boundary spring indices for strain
    n_boundary_springs = model.boundary_spring_count
    boundary_spring_indices = model.boundary_spring_indices
    
    print(f"\n{'='*60}")
    print(f"Running: {controller_type.upper()}")
    print(f"{'='*60}")
    print(f"  Model: {model.particle_count} particles, {n_boundary_springs} boundary springs")
    print(f"  Target pattern: {len(target_sequence)} steps")
    
    # =========================================================================
    # Build Nengo Network
    # =========================================================================
    
    # Shared state
    current_error = [0.0]
    current_strains = [np.zeros(n_boundary_springs, dtype=np.float32)]
    snn_output = [0.0]
    
    # Error callback
    error_scale = 0.5
    def get_error(t):
        return current_error[0] / error_scale
    
    # Create Nengo network
    nengo_model = nengo.Network(label=f"Inflation_{controller_type}")
    
    with nengo_model:
        if controller_type == 'snn_pid':
            controller = SNN_PID_Controller(
                Kp=1.0,
                Ki=0.5,
                Kd=0.3,
                u_max=10.0,
                n_neurons=n_neurons,
                error_scale=error_scale,
            )
            result = controller.build(
                model=nengo_model,
                get_error_callback=get_error,
            )
            
        else:  # snn_stress
            # Create strain input node for PES
            def get_strain(t):
                return current_strains[0]
            
            strain_input = nengo.Node(
                get_strain,
                size_out=n_boundary_springs,
                label="Strain_Input"
            )
            
            # Create strain ensemble
            strain_ensemble = nengo.Ensemble(
                n_neurons=n_neurons * min(n_boundary_springs, 10),
                dimensions=n_boundary_springs,
                max_rates=nengo.dists.Uniform(100, 200),
                intercepts=nengo.dists.Uniform(-0.5, 0.5),
                neuron_type=nengo.LIF(),
                radius=1.0,
                label="Boundary_Strain"
            )
            nengo.Connection(strain_input, strain_ensemble, synapse=0.01)
            
            controller = SNN_Stress_Controller(
                Kp=1.0,
                Kd=0.3,
                u_max=10.0,
                n_neurons=n_neurons,
                error_scale=error_scale,
                strain_dim=n_boundary_springs,
                pes_learning_rate=pes_learning_rate,
            )
            result = controller.build(
                model=nengo_model,
                get_error_callback=get_error,
                strain_ensemble=strain_ensemble,
            )
        
        # Output sync node
        def sync_output(t):
            snn_output[0] = controller.get_output()
        
        nengo.Node(sync_output, size_in=0, size_out=0, label="Output_Sync")
    
    # Create simulator
    sim = nengo.Simulator(nengo_model, dt=nengo_dt)
    
    # =========================================================================
    # Run Simulation
    # =========================================================================
    
    n_steps = len(target_sequence)
    physics_steps_per_nengo = int(dt / nengo_dt)
    
    # Data storage
    times = []
    volumes = []
    targets = []
    errors = []
    pressures = []
    strains_history = []
    
    last_physics_time = -dt
    sim_time = 0.0
    
    print(f"  Running {n_steps} physics steps...")
    
    for step in range(n_steps):
        target_ratio = target_sequence[step]
        
        # Update current volume and error
        current_volume = model.compute_current_volume(state)
        target_volume = model.initial_volume * target_ratio
        current_error[0] = target_volume - current_volume
        
        # Extract boundary strains
        if model.spring_strains is not None:
            all_strains = model.spring_strains.numpy()
            boundary_strains = all_strains[boundary_spring_indices]
            
            # Normalize to [-1, 1]
            strain_min = boundary_strains.min()
            strain_max = boundary_strains.max()
            strain_range = strain_max - strain_min
            if strain_range > 1e-8:
                normalized = 2.0 * (boundary_strains - strain_min) / strain_range - 1.0
            else:
                normalized = np.zeros_like(boundary_strains)
            current_strains[0] = normalized.astype(np.float32)
        
        # Run Nengo steps
        for _ in range(physics_steps_per_nengo):
            sim.step()
        
        # Get pressure from controller
        pressure = snn_output[0]
        
        # Apply inflation
        _, rest_config = model.apply_inflation(state, pressure, target_ratio)
        
        # Physics step
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
        
        # Store data
        sim_time += dt
        times.append(sim_time)
        volumes.append(current_volume / model.initial_volume)
        targets.append(target_ratio)
        errors.append(abs(target_volume - current_volume))
        pressures.append(pressure)
        strains_history.append(np.mean(np.abs(current_strains[0])))
        
        # Progress
        if step % 100 == 0:
            vol_ratio = current_volume / model.initial_volume
            print(f"    Step {step:4d}: vol={vol_ratio:.3f} target={target_ratio:.3f} err={errors[-1]:.4f} p={pressure:.2f}")
    
    sim.close()
    
    return {
        'times': np.array(times),
        'volumes': np.array(volumes),
        'targets': np.array(targets),
        'errors': np.array(errors),
        'pressures': np.array(pressures),
        'strains': np.array(strains_history),
    }


def run_comparison(args):
    """Run comparison between SNN_PID and SNN_Stress."""
    
    total_time = args.total_time
    dt = args.dt
    n_steps = int(total_time / dt)
    
    print("\n" + "=" * 70)
    print("  CONTROLLER COMPARISON: SNN_PID vs SNN_Stress")
    print("=" * 70)
    print(f"  Total time: {total_time}s ({n_steps} steps)")
    print(f"  Target pattern: {args.pattern}")
    print(f"  Max volume: {args.max_volume}x")
    print(f"  PES learning rate: {args.pes_lr}")
    
    # Create target sequence
    target_sequence = create_target_sequence(
        total_time=total_time,
        dt=dt,
        max_volume=args.max_volume,
        pattern=args.pattern,
    )
    
    # Run SNN_PID
    print("\n" + "-" * 40)
    pid_data = run_single_simulation(
        controller_type='snn_pid',
        target_sequence=target_sequence,
        dt=dt,
        nengo_dt=args.nengo_dt,
        max_volume=args.max_volume,
        n_neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    
    # Run SNN_Stress
    print("\n" + "-" * 40)
    stress_data = run_single_simulation(
        controller_type='snn_stress',
        target_sequence=target_sequence,
        dt=dt,
        nengo_dt=args.nengo_dt,
        max_volume=args.max_volume,
        pes_learning_rate=args.pes_lr,
        n_neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"comparison_{timestamp}.npz"
    
    np.savez(
        output_file,
        pid_times=pid_data['times'],
        pid_volumes=pid_data['volumes'],
        pid_targets=pid_data['targets'],
        pid_errors=pid_data['errors'],
        pid_pressures=pid_data['pressures'],
        pid_strains=pid_data['strains'],
        stress_times=stress_data['times'],
        stress_volumes=stress_data['volumes'],
        stress_targets=stress_data['targets'],
        stress_errors=stress_data['errors'],
        stress_pressures=stress_data['pressures'],
        stress_strains=stress_data['strains'],
        dt=dt,
        total_time=total_time,
        pattern=args.pattern,
        max_volume=args.max_volume,
        pes_lr=args.pes_lr,
    )
    print(f"\nResults saved: {output_file}")
    
    # Plot
    if not args.no_plot:
        plot_comparison(output_file, args)
    
    return output_file


def plot_comparison(data_file, args):
    """Plot comparison results from saved .npz file."""
    
    print(f"\nPlotting: {data_file}")
    data = np.load(data_file, allow_pickle=True)
    
    # Extract data
    pid_times = data['pid_times']
    pid_volumes = data['pid_volumes']
    pid_targets = data['pid_targets']
    pid_errors = data['pid_errors']
    pid_pressures = data['pid_pressures']
    
    stress_times = data['stress_times']
    stress_volumes = data['stress_volumes']
    stress_targets = data['stress_targets']
    stress_errors = data['stress_errors']
    stress_pressures = data['stress_pressures']
    
    total_time = float(data['total_time'])
    pes_lr = float(data['pes_lr'])
    
    # Compute early vs late performance
    split_idx = len(pid_times) // 2
    
    pid_early = np.sqrt(np.mean(pid_errors[:split_idx]**2))
    pid_late = np.sqrt(np.mean(pid_errors[split_idx:]**2))
    stress_early = np.sqrt(np.mean(stress_errors[:split_idx]**2))
    stress_late = np.sqrt(np.mean(stress_errors[split_idx:]**2))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTracking Error (RMS):")
    print(f"  {'SNN_PID':12s}: Early={pid_early:.4f}, Late={pid_late:.4f}")
    print(f"  {'SNN_Stress':12s}: Early={stress_early:.4f}, Late={stress_late:.4f}")
    
    if pid_late > 0:
        improvement = (1 - stress_late/pid_late) * 100
        print(f"\n  SNN_Stress improvement (late phase): {improvement:+.1f}%")
    
    # Learning effect
    pid_learning = (pid_late - pid_early) / pid_early * 100 if pid_early > 0 else 0
    stress_learning = (stress_early - stress_late) / stress_early * 100 if stress_early > 0 else 0
    print(f"\n  PES Learning effect:")
    print(f"    SNN_PID error change: {pid_learning:+.1f}% (no learning expected)")
    print(f"    SNN_Stress improvement: {stress_learning:+.1f}% (PES learning!)")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SNN_PID vs SNN_Stress (PES lr={pes_lr})\nDemonstrating PES Feedforward Learning', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Volume tracking
    ax = axes[0, 0]
    ax.plot(pid_times, pid_targets, 'k--', label='Target', linewidth=2, alpha=0.7)
    ax.plot(pid_times, pid_volumes, 'b-', label='SNN_PID', alpha=0.8)
    ax.plot(stress_times, stress_volumes, 'r-', label='SNN_Stress', alpha=0.8)
    ax.axvline(pid_times[split_idx], color='gray', linestyle='--', alpha=0.5, label='Half-time')
    ax.fill_between(pid_times[:split_idx], 0.9, 2.1, alpha=0.1, color='yellow', label='Early (learning)')
    ax.fill_between(pid_times[split_idx:], 0.9, 2.1, alpha=0.1, color='green', label='Late (adapted)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Volume Ratio')
    ax.set_title('Volume Tracking')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Tracking error over time
    ax = axes[0, 1]
    ax.plot(pid_times, pid_errors, 'b-', label='SNN_PID', alpha=0.8)
    ax.plot(stress_times, stress_errors, 'r-', label='SNN_Stress', alpha=0.8)
    ax.axvline(pid_times[split_idx], color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(pid_times[:split_idx], ax.get_ylim()[0], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, 
                   alpha=0.1, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error')
    ax.set_title('Tracking Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Early vs Late comparison (bar chart)
    ax = axes[1, 0]
    x_pos = np.array([0, 1])
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, [pid_early, pid_late], width, 
                   label='SNN_PID', color='blue', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, [stress_early, stress_late], width, 
                   label='SNN_Stress', color='red', alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('RMS Error')
    ax.set_title('Early vs Late Performance\n(PES should improve late phase)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Early\n(learning)', 'Late\n(adapted)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Control pressure
    ax = axes[1, 1]
    ax.plot(pid_times, pid_pressures, 'b-', label='SNN_PID', alpha=0.7)
    ax.plot(stress_times, stress_pressures, 'r-', label='SNN_Stress', alpha=0.7)
    ax.axvline(pid_times[split_idx], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure')
    ax.set_title('Control Pressure (PES adds feedforward component)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = data_file.replace('.npz', '.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    if args.show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare SNN_PID vs SNN_Stress Controllers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_controllers.py                      # Default comparison
  python compare_controllers.py --total-time 30      # Longer simulation
  python compare_controllers.py --pes-lr 1e-3        # Higher learning rate
  python compare_controllers.py --pattern multi_step # Challenge with multiple steps
  python compare_controllers.py --plot-file X.npz    # Plot from saved data
        """
    )
    
    # Simulation parameters
    parser.add_argument('--total-time', '-t', type=float, default=20.0,
                       help='Total simulation time in seconds (default: 20)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Physics time step (default: 0.01)')
    parser.add_argument('--nengo-dt', type=float, default=0.001,
                       help='Nengo time step (default: 0.001)')
    
    # Target pattern
    parser.add_argument('--pattern', type=str, default='multi_step',
                       choices=['step', 'ramp', 'sine', 'multi_step'],
                       help='Target pattern (default: multi_step)')
    parser.add_argument('--max-volume', type=float, default=2.0,
                       help='Maximum volume ratio (default: 2.0)')
    
    # Controller parameters
    parser.add_argument('--pes-lr', type=float, default=1e-4,
                       help='PES learning rate for SNN_Stress (default: 1e-4)')
    parser.add_argument('--neurons', type=int, default=100,
                       help='Neurons per ensemble (default: 100)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Compute device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename (.npz)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')
    parser.add_argument('--plot-file', type=str, default=None,
                       help='Plot from existing .npz file (skip simulation)')
    
    args = parser.parse_args()
    
    if args.plot_file:
        plot_comparison(args.plot_file, args)
    else:
        run_comparison(args)


if __name__ == '__main__':
    main()
