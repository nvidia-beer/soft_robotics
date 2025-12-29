#!/usr/bin/env python3
"""
Trajectory Tracking Simulation

Run trajectory tracking with different controllers:
- pid: Classical PID
- mpc: Classical MPC  
- stress: Classical stress-adaptive control (strain-based modulation)
- snn_pid: NEF-based spiking PD (PURE - from Zaidel et al. 2021)
- snn_stress: NEF-based spiking stress-adaptive (with strain feedback)
- snn_mpc: MPC + SNN adaptation

Usage:
    python run_tracking.py                         # Default (mpc)
    python run_tracking.py --controller pid        # Use PID
    python run_tracking.py --controller stress     # Stress-adaptive control
    python run_tracking.py --controller snn_pid    # NEF spiking PID
    python run_tracking.py --controller snn_stress # NEF stress-adaptive
    python run_tracking.py --controller snn_mpc --no-display
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime


def create_controller(name, env, args):
    """Create controller by name."""
    from controllers import MPC, PID, Stress, NengoMPC, NengoPID, NengoStress, NENGO_AVAILABLE
    
    num_groups = env.num_groups
    
    if name == 'pid':
        return PID(
            num_groups=num_groups,
            dt=args.dt,
            u_max=args.u_max,
            Kp=args.kp,
            Ki=args.ki,
            Kd=args.kd,
        ), "PID"
    
    elif name == 'mpc':
        return MPC(
            num_groups=num_groups,
            dt=args.dt,
            u_max=args.u_max,
            horizon=args.horizon,
            spring_k=args.model_k,
            damping=args.model_c,
            Q=args.Q,
            R=args.R,
        ), "MPC"
    
    elif name == 'stress':
        return Stress(
            num_groups=num_groups,
            dt=args.dt,
            u_max=args.u_max,
            Kp=args.kp,
            Kd=args.kd,
            alpha=args.stress_alpha,
            beta=args.stress_beta,
        ), "Stress"
    
    elif name == 'snn_pid':
        if not NENGO_AVAILABLE:
            raise ImportError("Nengo not available for SNN controllers")
        return NengoPID(
            num_groups=num_groups,
            dt=args.nengo_dt,  # Finer dt for smooth spike decoding
            u_max=args.u_max,
            Kp=args.kp,
            Ki=0.0,  # Pure PD
            Kd=args.kd,
            n_neurons=args.snn_neurons,
            device=args.device,
        ), "SNN-PID (PURE)"
    
    elif name == 'snn_stress':
        if not NENGO_AVAILABLE:
            raise ImportError("Nengo not available for SNN controllers")
        return NengoStress(
            num_groups=num_groups,
            dt=args.nengo_dt,  # Finer dt for smooth spike decoding
            u_max=args.u_max,
            Kp=args.kp,
            Kd=args.kd,
            n_neurons=args.snn_neurons,
            pes_learning_rate=args.pes_learning_rate,
            device=args.device,
        ), "SNN-Stress"
    
    elif name == 'snn_mpc':
        if not NENGO_AVAILABLE:
            raise ImportError("Nengo not available for SNN controllers")
        return NengoMPC(
            num_groups=num_groups,
            dt=args.nengo_dt,  # Finer dt for smooth spike decoding
            u_max=args.u_max,
            horizon=args.horizon,
            spring_k=args.model_k,
            damping=args.model_c,
            Q=args.Q,
            R=args.R,
            snn_neurons=args.snn_neurons,
            snn_learning_rate=args.snn_lr,
            snn_tau=args.snn_tau,
            learning_enabled=not args.no_learning,
        ), "MPC+SNN" if not args.no_learning else "MPC+SNN (no learning)"
    
    else:
        raise ValueError(f"Unknown controller: {name}")


def run_simulation(args):
    """Run single controller simulation."""
    from tracking_env import TrackingEnv
    from tqdm import tqdm
    
    print("=" * 60)
    print("TRAJECTORY TRACKING SIMULATION")
    print("  MODE: RIGID GRID (all group centroids)")
    print("=" * 60)
    
    # Calculate steps from time or use direct step count
    if args.total_time is not None:
        n_steps = int(args.total_time / args.dt)
    else:
        n_steps = args.total_steps
    render_mode = None if args.no_display else 'human'
    
    # Create environment
    env = TrackingEnv(
        render_mode=render_mode,
        N=args.grid_size,
        dt=args.dt,
        spring_stiffness=args.spring_k,
        spring_damping=args.spring_c,
        trajectory_type=args.trajectory,
        trajectory_amplitude=args.amplitude,
        trajectory_frequency=args.frequency,
        device=args.device,
        use_fem=True,
        boxsize=2.5,
        window_width=1200,
        window_height=800,
    )
    env.max_time = args.total_time  # For plot scaling
    
    # Create controller
    ctrl, ctrl_name = create_controller(args.controller, env, args)
    
    
    print(f"\nController: {ctrl_name}")
    print(f"Trajectory: {args.trajectory} (A={args.amplitude}, f={args.frequency}Hz)")
    print(f"Duration: {n_steps * args.dt:.1f}s ({n_steps} steps)")
    
    print(f"\nStructural Mismatch (by design):")
    print(f"  Reality (FEM):      Nonlinear hyperelastic")
    print(f"  MPC model (simple): k={args.model_k}, c={args.model_c}")
    wind_mag = np.linalg.norm(args.wind)
    if wind_mag > 0:
        print(f"\nUNKNOWN WIND FORCE (model does NOT know!):")
        print(f"  Force: [{args.wind[0]}, {args.wind[1]}]")
    
    # Set wind force for visualization
    env.set_wind(args.wind)
    
    # Run simulation (use seed for reproducibility)
    obs, _ = env.reset(seed=args.seed)
    ctrl.reset()
    
    errors = []
    targets = []
    centroids = []
    times = []
    
    # Progress bar
    pbar = tqdm(range(n_steps), desc=f"  {ctrl_name}", unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for step in pbar:
        state = env.get_state_for_controller()
        u = ctrl.compute_control(state, env.get_target_position)
        
        # Add UNKNOWN wind force (like paper's steering-malfunction)
        # The model does NOT know about this - SNN must learn it!
        wind_force = np.array(args.wind)
        if np.linalg.norm(wind_force) > 0:
            for i in range(len(env.center_indices)):
                u[i*2:i*2+2] += wind_force
        
        obs, reward, done, trunc, info = env.step(u)
        
        errors.append(info['tracking_error'])
        targets.append(info['target'].copy())
        init_center = np.mean(env.initial_positions[env.center_indices], axis=0)
        centroids.append(env.get_center_centroid() - init_center)
        times.append(step * args.dt)
        
        # Update progress bar with current error
        pbar.set_postfix({'error': f'{info["tracking_error"]:.4f}'})
        
        if render_mode:
            env.render()
    
    env.close()
    
    # Results
    errors = np.array(errors)
    times = np.array(times)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Max error:  {np.max(errors):.4f}")
    print(f"  Final error: {errors[-1]:.4f}")
    
    # Plot
    if not args.no_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Trajectory Tracking: {ctrl_name}', fontsize=14)
        
        # Error over time
        axes[0, 0].plot(times, errors, 'b-', linewidth=1)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Tracking Error')
        axes[0, 0].set_title('Tracking Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X trajectory
        target_x = [t[0] for t in targets]
        centroid_x = [c[0] for c in centroids]
        axes[0, 1].plot(times, target_x, 'k--', label='Target', linewidth=2)
        axes[0, 1].plot(times, centroid_x, 'b-', label='Actual', alpha=0.8)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('X Position')
        axes[0, 1].set_title('X Component')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Y trajectory
        target_y = [t[1] for t in targets]
        centroid_y = [c[1] for c in centroids]
        axes[1, 0].plot(times, target_y, 'k--', label='Target', linewidth=2)
        axes[1, 0].plot(times, centroid_y, 'b-', label='Actual', alpha=0.8)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].set_title('Y Component')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # XY trajectory
        axes[1, 1].plot([t[0] for t in targets], [t[1] for t in targets], 
                       'k--', label='Target', linewidth=2)
        axes[1, 1].plot([c[0] for c in centroids], [c[1] for c in centroids], 
                       'b-', label='Actual', alpha=0.8)
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].set_title('XY Trajectory')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        
        plot_file = f'tracking_{args.controller}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {plot_file}")
        
        if not args.no_display:
            plt.show()
        plt.close()


def run_comparison(args):
    """Run comparison between classical and SNN controller."""
    from tracking_env import TrackingEnv
    from controllers import MPC, PID, NengoMPC, NengoPID, NENGO_AVAILABLE
    from tqdm import tqdm
    
    if not NENGO_AVAILABLE:
        raise ImportError("Nengo required for comparison mode")
    
    # Determine which comparison
    if args.controller in ['pid', 'snn_pid']:
        classical_name = 'pid'
        snn_name = 'snn_pid'
        title = 'PID vs SNN-PID'
    else:
        classical_name = 'mpc'
        snn_name = 'snn_mpc'
        title = 'MPC vs MPC+SNN'
    
    print("=" * 60)
    print(f"COMPARISON: {title}")
    print("=" * 60)
    print(f"  Seed: {args.seed} (ensures identical initial conditions)")
    wind_mag = np.linalg.norm(args.wind)
    if wind_mag > 0:
        print(f"  WIND: [{args.wind[0]}, {args.wind[1]}] (model does NOT know!)")
    
    # Calculate steps from time or use direct step count
    if args.total_time is not None:
        n_steps = int(args.total_time / args.dt)
    else:
        n_steps = args.total_steps
    render_mode = None if args.no_display else 'human'
    
    results = {}
    
    for ctrl_name in [snn_name, classical_name]:  # SNN first so we can see learning
        print(f"\n--- Running: {ctrl_name.upper()} ---")
        
        env = TrackingEnv(
            render_mode=render_mode,
            N=args.grid_size,
            dt=args.dt,
            spring_stiffness=args.spring_k,
            spring_damping=args.spring_c,
            trajectory_type=args.trajectory,
            trajectory_amplitude=args.amplitude,
            trajectory_frequency=args.frequency,
            device=args.device,
            use_fem=True,
            boxsize=2.5,
            window_width=1200,
            window_height=800,
        )
        env.max_time = args.total_time
        
        ctrl, label = create_controller(ctrl_name, env, args)
        
        
        # Set wind force for visualization
        env.set_wind(args.wind)
        
        # Use SAME seed for both controllers = identical initial conditions
        obs, _ = env.reset(seed=args.seed)
        ctrl.reset()
        
        errors = []
        targets = []
        centroids = []
        
        # Progress bar for this controller
        pbar = tqdm(range(n_steps), desc=f"  {label}", unit="step", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for step in pbar:
            state = env.get_state_for_controller()
            u = ctrl.compute_control(state, env.get_target_position)
            
            # Add UNKNOWN wind force (like paper's steering-malfunction)
            # The model does NOT know about this - SNN must learn it!
            wind_force = np.array(args.wind)
            if np.linalg.norm(wind_force) > 0:
                for i in range(len(env.center_indices)):
                    u[i*2:i*2+2] += wind_force
            
            obs, reward, done, trunc, info = env.step(u)
            
            errors.append(info['tracking_error'])
            targets.append(info['target'].copy())
            init_center = np.mean(env.initial_positions[env.center_indices], axis=0)
            centroids.append(env.get_center_centroid() - init_center)
            
            # Update progress bar with current error
            pbar.set_postfix({'error': f'{info["tracking_error"]:.4f}'})
            
            if render_mode:
                env.render()
        
        env.close()
        
        results[ctrl_name] = {
            'errors': np.array(errors),
            'targets': targets,
            'centroids': centroids,
            'label': label
        }
    
    # Save results to file
    times = np.arange(n_steps) * args.dt
    
    output_file = args.output if args.output else f'{classical_name}_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz'
    
    save_data = {
        'times': times,
        'dt': args.dt,
        'total_time': args.total_time,
        'seed': args.seed,
        'wind': np.array(args.wind),
        'classical_name': classical_name,
        'snn_name': snn_name,
        'classical_label': results[classical_name]['label'],
        'snn_label': results[snn_name]['label'],
        'classical_errors': results[classical_name]['errors'],
        'snn_errors': results[snn_name]['errors'],
        'classical_targets': np.array(results[classical_name]['targets']),
        'snn_targets': np.array(results[snn_name]['targets']),
        'classical_centroids': np.array(results[classical_name]['centroids']),
        'snn_centroids': np.array(results[snn_name]['centroids']),
    }
    
    np.savez(output_file, **save_data)
    print(f"\nResults saved: {output_file}")
    
    # Plot from saved data
    if not args.no_plot:
        plot_comparison(output_file, args)


def run_triple_comparison(args):
    """Run 3-way comparison: PID vs SNN-PID vs SNN-Stress."""
    from tracking_env import TrackingEnv
    from controllers import PID, NENGO_AVAILABLE
    from tqdm import tqdm
    
    if not NENGO_AVAILABLE:
        raise ImportError("Nengo required for comparison mode")
    
    print("=" * 70)
    print("3-WAY COMPARISON: PID vs SNN-PID vs SNN-Stress")
    print("=" * 70)
    print(f"  Grid: {args.grid_size}x{args.grid_size} ({(args.grid_size-1)**2} groups)")
    print(f"  Seed: {args.seed} (ensures identical initial conditions)")
    print(f"  FEM: Implicit solver (Neo-Hookean hyperelastic)")
    wind_mag = np.linalg.norm(args.wind)
    if wind_mag > 0:
        print(f"  WIND: [{args.wind[0]}, {args.wind[1]}] (model does NOT know!)")
    
    # Calculate steps
    if args.total_time is not None:
        n_steps = int(args.total_time / args.dt)
    else:
        n_steps = args.total_steps
    render_mode = None if args.no_display else 'human'
    
    # Controllers to compare (SNN-Stress first to show learning, then baselines)
    controllers_to_run = ['snn_stress', 'pid', 'snn_pid']
    results = {}
    
    for ctrl_name in controllers_to_run:
        print(f"\n{'='*50}")
        print(f"Running: {ctrl_name.upper()}")
        print(f"{'='*50}")
        
        env = TrackingEnv(
            render_mode=render_mode,
            N=args.grid_size,
            dt=args.dt,
            spring_stiffness=args.spring_k,
            spring_damping=args.spring_c,
            trajectory_type=args.trajectory,
            trajectory_amplitude=args.amplitude,
            trajectory_frequency=args.frequency,
            device=args.device,
            use_fem=True,
            boxsize=2.5,
            window_width=1200,
            window_height=800,
        )
        env.max_time = args.total_time
        
        ctrl, label = create_controller(ctrl_name, env, args)
        
        # Set wind force
        env.set_wind(args.wind)
        
        # Use SAME seed for all controllers = identical initial conditions
        obs, _ = env.reset(seed=args.seed)
        ctrl.reset()
        
        errors = []
        targets = []
        centroids = []
        
        pbar = tqdm(range(n_steps), desc=f"  {label}", unit="step",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for step in pbar:
            state = env.get_state_for_controller()
            u = ctrl.compute_control(state, env.get_target_position)
            
            # Add UNKNOWN wind force
            wind_force = np.array(args.wind)
            if np.linalg.norm(wind_force) > 0:
                for i in range(len(env.center_indices)):
                    u[i*2:i*2+2] += wind_force
            
            obs, reward, done, trunc, info = env.step(u)
            
            errors.append(info['tracking_error'])
            targets.append(info['target'].copy())
            init_center = np.mean(env.initial_positions[env.center_indices], axis=0)
            centroids.append(env.get_center_centroid() - init_center)
            
            pbar.set_postfix({'error': f'{info["tracking_error"]:.4f}'})
            
            if render_mode:
                env.render()
        
        # Save final frame screenshot (before closing window)
        screenshot_path = None
        if render_mode and env.window is not None:
            import pygame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"comparison_{ctrl_name}_{timestamp}.png"
            pygame.image.save(env.window, screenshot_path)
            print(f"  Saved final frame: {screenshot_path}")
        
        env.close()
        
        results[ctrl_name] = {
            'errors': np.array(errors),
            'targets': targets,
            'centroids': centroids,
            'label': label,
            'screenshot': screenshot_path
        }
    
    # Save results
    times = np.arange(n_steps) * args.dt
    output_file = args.output if args.output else f'triple_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz'
    
    save_data = {
        'times': times,
        'dt': args.dt,
        'total_time': args.total_time,
        'seed': args.seed,
        'wind': np.array(args.wind),
        'grid_size': args.grid_size,
    }
    
    for ctrl_name in controllers_to_run:
        save_data[f'{ctrl_name}_errors'] = results[ctrl_name]['errors']
        save_data[f'{ctrl_name}_targets'] = np.array(results[ctrl_name]['targets'])
        save_data[f'{ctrl_name}_centroids'] = np.array(results[ctrl_name]['centroids'])
        save_data[f'{ctrl_name}_label'] = results[ctrl_name]['label']
        if 'screenshot' in results[ctrl_name]:
            save_data[f'{ctrl_name}_screenshot'] = results[ctrl_name]['screenshot']
    
    np.savez(output_file, **save_data)
    print(f"\nResults saved: {output_file}")
    
    # Plot results
    if not args.no_plot:
        plot_triple_comparison(output_file, args)


def plot_triple_comparison(data_file, args=None):
    """Plot 3-way comparison: PID vs SNN-PID vs SNN-Stress."""
    data = np.load(data_file, allow_pickle=True)
    
    times = data['times']
    controllers = ['pid', 'snn_pid', 'snn_stress']
    colors = {'pid': 'blue', 'snn_pid': 'orange', 'snn_stress': 'green'}
    
    figsize = getattr(args, 'figsize', (16, 12)) if args else (16, 12)
    dpi = getattr(args, 'dpi', 150) if args else 150
    show = not getattr(args, 'no_display', False) if args else True
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'3-Way Comparison: PID vs SNN-PID vs SNN-Stress (Grid {data["grid_size"]}x{data["grid_size"]})',
                 fontsize=14, fontweight='bold')
    
    # Analysis
    split_idx = len(times) // 2
    early_rms = {}
    late_rms = {}
    
    print("\n" + "=" * 70)
    print("RESULTS: 3-WAY COMPARISON")
    print("=" * 70)
    
    for ctrl_name in controllers:
        errors = data[f'{ctrl_name}_errors']
        label = str(data[f'{ctrl_name}_label'])
        early_rms[ctrl_name] = np.sqrt(np.mean(errors[:split_idx]**2))
        late_rms[ctrl_name] = np.sqrt(np.mean(errors[split_idx:]**2))
        
        print(f"  {label:20s}: Early={early_rms[ctrl_name]:.4f}, Late={late_rms[ctrl_name]:.4f}")
        
        # Error plot
        axes[0, 0].plot(times, errors, color=colors[ctrl_name], label=label, alpha=0.8)
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Tracking Error')
    axes[0, 0].set_title('Tracking Error Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=times[split_idx], color='gray', linestyle='--', alpha=0.5, label='Split')
    
    # X trajectory
    for ctrl_name in controllers:
        centroids = data[f'{ctrl_name}_centroids']
        label = str(data[f'{ctrl_name}_label'])
        axes[0, 1].plot(times, centroids[:, 0], color=colors[ctrl_name], label=label, alpha=0.8)
    
    targets = data[f'{controllers[0]}_targets']
    axes[0, 1].plot(times, targets[:, 0], 'k--', label='Target', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position')
    axes[0, 1].set_title('X Component Tracking')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bar comparison
    x_pos = np.arange(len(controllers))
    width = 0.35
    
    early_vals = [early_rms[c] for c in controllers]
    late_vals = [late_rms[c] for c in controllers]
    labels = [str(data[f'{c}_label']) for c in controllers]
    bar_colors = [colors[c] for c in controllers]
    
    axes[1, 0].bar(x_pos - width/2, early_vals, width, label='Early (learning)', alpha=0.7,
                   color=bar_colors)
    axes[1, 0].bar(x_pos + width/2, late_vals, width, label='Late (adapted)', alpha=0.9,
                   color=bar_colors)
    axes[1, 0].set_ylabel('RMS Error')
    axes[1, 0].set_title('Early vs Late Performance')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Improvement percentages
    print(f"\nImprovement (Late vs Early):")
    improvements = []
    for ctrl_name in controllers:
        if early_rms[ctrl_name] > 0:
            imp = (early_rms[ctrl_name] - late_rms[ctrl_name]) / early_rms[ctrl_name] * 100
            improvements.append(imp)
            print(f"  {str(data[f'{ctrl_name}_label']):20s}: {imp:+.1f}%")
        else:
            improvements.append(0)
    
    axes[1, 1].bar(x_pos, improvements, color=bar_colors, alpha=0.8)
    axes[1, 1].set_ylabel('Improvement %')
    axes[1, 1].set_title('Learning Improvement (Early→Late)')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_file = data_file.replace('.npz', '.png')
    plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    if show:
        plt.show()
    plt.close()


def plot_comparison(data_file, args=None):
    """
    Plot comparison results from a saved .npz file.
    
    Args:
        data_file: Path to .npz file with comparison results
        args: Optional args with visualization parameters (figsize, dpi, title, etc.)
    """
    # Load data
    data = np.load(data_file, allow_pickle=True)
    
    times = data['times']
    classical_errors = data['classical_errors']
    snn_errors = data['snn_errors']
    classical_targets = data['classical_targets']
    snn_targets = data['snn_targets']
    classical_centroids = data['classical_centroids']
    snn_centroids = data['snn_centroids']
    classical_label = str(data['classical_label'])
    snn_label = str(data['snn_label'])
    classical_name = str(data['classical_name'])
    wind = data['wind']
    
    # Visualization parameters
    figsize = getattr(args, 'figsize', (14, 10)) if args else (14, 10)
    dpi = getattr(args, 'dpi', 150) if args else 150
    title = getattr(args, 'title', None) if args else None
    show = not getattr(args, 'no_display', False) if args else True
    
    # Analysis
    split_idx = len(times) // 2
    classical_early = np.sqrt(np.mean(classical_errors[:split_idx]**2))
    classical_late = np.sqrt(np.mean(classical_errors[split_idx:]**2))
    snn_early = np.sqrt(np.mean(snn_errors[:split_idx]**2))
    snn_late = np.sqrt(np.mean(snn_errors[split_idx:]**2))
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTracking Error (RMS):")
    print(f"  {classical_label:10s}: Early={classical_early:.4f}, Late={classical_late:.4f}")
    print(f"  {snn_label:10s}: Early={snn_early:.4f}, Late={snn_late:.4f}")
    
    if classical_late > 0:
        improvement = (1 - snn_late/classical_late) * 100
        print(f"\n  SNN improvement (late phase): {improvement:+.1f}%")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=14)
    else:
        wind_str = f", Wind=[{wind[0]:.1f},{wind[1]:.1f}]" if np.linalg.norm(wind) > 0 else ""
        fig.suptitle(f'{classical_label} vs {snn_label} (Simple Model vs FEM Reality{wind_str})', fontsize=14)
    
    # Tracking error
    axes[0, 0].plot(times, classical_errors, 'b-', label=classical_label, alpha=0.8)
    axes[0, 0].plot(times, snn_errors, 'r-', label=snn_label, alpha=0.8)
    axes[0, 0].axvline(times[split_idx], color='gray', linestyle='--', alpha=0.5, label='Half-time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Tracking Error')
    axes[0, 0].set_title('Tracking Error Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # X trajectory
    target_x = classical_targets[:, 0]
    classical_x = classical_centroids[:, 0]
    snn_x = snn_centroids[:, 0]
    
    axes[0, 1].plot(times, target_x, 'k--', label='Target', linewidth=2)
    axes[0, 1].plot(times, classical_x, 'b-', label=classical_label, alpha=0.8)
    axes[0, 1].plot(times, snn_x, 'r-', label=snn_label, alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position')
    axes[0, 1].set_title('X Component')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bar comparison
    x_pos = np.array([0, 1])
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, [classical_early, classical_late], width, 
                  label=classical_label, color='blue', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, [snn_early, snn_late], width, 
                  label=snn_label, color='red', alpha=0.7)
    axes[1, 0].set_ylabel('RMS Error')
    axes[1, 0].set_title('Early vs Late Performance')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(['Early\n(learning)', 'Late\n(adapted)'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # XY trajectory
    axes[1, 1].plot(classical_targets[:, 0], classical_targets[:, 1], 
                   'k--', label='Target', linewidth=2)
    axes[1, 1].plot(classical_centroids[:, 0], classical_centroids[:, 1], 
                   'b-', label=classical_label, alpha=0.7)
    axes[1, 1].plot(snn_centroids[:, 0], snn_centroids[:, 1], 
                   'r-', label=snn_label, alpha=0.7)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('XY Trajectory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    
    plot_file = data_file.replace('.npz', '.png')
    plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Trajectory Tracking Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tracking.py                          # Default MPC
  python run_tracking.py --controller pid         # Use PID
  python run_tracking.py --controller snn_mpc     # MPC + SNN
  python run_tracking.py --compare                # Compare MPC vs MPC+SNN
  
  # Save results and plot later:
  python run_tracking.py --compare --no-plot      # Just save results to .npz
  python run_tracking.py --plot-file mpc_comparison_*.npz  # Plot from saved file

RIGID GRID MODE:
  - Controls ALL group centroids with rotation
  - Each group centroid tracks its own target = initial_pos + offset + rotation
  - The grid moves and rotates as a rigid body
  - FEM deformation resists rigidity -> complex control problem
  - SNN learns the model-world mismatch
        """
    )
    
    # Controller selection
    parser.add_argument('--controller', '-c', type=str, default='mpc',
                       choices=['pid', 'mpc', 'stress', 'snn_pid', 'snn_stress', 'snn_mpc'],
                       help='Controller type')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison (classical vs SNN)')
    parser.add_argument('--compare3', action='store_true',
                       help='Run 3-way comparison: PID vs SNN-PID vs SNN-Stress')
    
    # Environment
    parser.add_argument('--grid-size', '-N', type=int, default=3,
                       help='Grid size N>=2: N=2→1 group, N=3→4 groups, N=4→9 groups')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (physics dt)')
    parser.add_argument('--nengo-dt', type=float, default=0.001,
                       help='Nengo dt (finer for smooth spike decoding, default 0.001s)')
    parser.add_argument('--total-time', '-t', type=float, default=None,
                       help='Total simulation time in seconds')
    parser.add_argument('--total-steps', type=int, default=500,
                       help='Total simulation steps (default: 500)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Compute device (cuda or cpu)')
    
    # Trajectory - faster to challenge the controller
    parser.add_argument('--trajectory', type=str, default='circular',
                       choices=['sinusoidal', 'circular', 'figure8'])
    parser.add_argument('--amplitude', '-A', type=float, default=0.3)
    parser.add_argument('--frequency', '-f', type=float, default=0.5,
                       help='Higher = faster trajectory = harder to track')
    
    # FEM simulation parameters (real physics)
    parser.add_argument('--spring-k', type=float, default=40.0,
                       help='FEM material stiffness')
    parser.add_argument('--spring-c', type=float, default=0.5,
                       help='FEM material damping')
    
    # MPC simplified model parameters - VERY WRONG to show SNN benefit
    parser.add_argument('--model-k', type=float, default=40.0,
                       help='Model stiffness (matches FEM, simplified: no mesh coupling)')
    parser.add_argument('--model-c', type=float, default=0.5,
                       help='Model damping (matches FEM)')
    
    # PID gains
    parser.add_argument('--kp', type=float, default=200.0)
    parser.add_argument('--ki', type=float, default=10.0)
    parser.add_argument('--kd', type=float, default=50.0)
    
    # MPC parameters
    parser.add_argument('--horizon', type=int, default=10,
                       help='Prediction horizon (5=fast, 10=balanced, 20=accurate)')
    parser.add_argument('--Q', type=float, default=500.0,
                       help='Tracking error weight (higher = more aggressive)')
    parser.add_argument('--R', type=float, default=0.001,
                       help='Control effort weight (lower = more aggressive)')
    parser.add_argument('--u-max', type=float, default=500.0,
                       help='Max control force (same for both MPCs)')
    
    # SNN parameters - Tuned for trajectory tracking
    parser.add_argument('--snn-neurons', type=int, default=500,
                       help='Neurons per dimension (500 = smooth output)')
    parser.add_argument('--snn-lr', type=float, default=1e-3,
                       help='Learning rate for PES (higher = faster learning)')
    parser.add_argument('--snn-tau', type=float, default=0.05,
                       help='Synapse time constant (paper default: 0.05)')
    parser.add_argument('--no-learning', action='store_true',
                       help='Disable SNN learning (NengoMPC runs identical to vanilla MPC)')
    
    # Stress control parameters (for classic stress controller)
    parser.add_argument('--stress-alpha', type=float, default=0.5,
                       help='Compliance factor (0=stiff, 1=max compliance at max strain)')
    parser.add_argument('--stress-beta', type=float, default=20.0,
                       help='Strain-rate damping coefficient (N·s per unit strain/s)')
    
    # PES learning rate (for snn_stress)
    parser.add_argument('--pes-learning-rate', type=float, default=1e-4,
                       help='PES feedforward learning rate (0 to disable)')
    
    # Unknown disturbance (like paper's --steering-malfunction)
    # Tune empirically: start small, increase until SNN shows benefit
    # Default to no wind - FEM model mismatch alone is sufficient for SNN learning
    parser.add_argument('--wind', type=float, nargs=2, default=[0.0, 0.0],
                       help='Constant wind force [fx, fy] that model does NOT know about')
    
    # Display
    parser.add_argument('--no-display', action='store_true',
                       help='Disable pygame display')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable matplotlib plots (still saves .npz)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducible comparisons')
    
    # Results file options
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename for results (.npz)')
    parser.add_argument('--plot-file', type=str, default=None,
                       help='Plot from saved .npz file (skip simulation)')
    
    # Visualization parameters (for --plot-file)
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 10],
                       help='Figure size [width, height]')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Plot DPI')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom plot title')
    
    args = parser.parse_args()
    
    # Validate grid size: must be even and >= 4 (to have a center group)
    if args.grid_size < 2:
        parser.error(f"Grid size must be >= 2 (got {args.grid_size}). "
                    f"Need at least 2x2 particles for 1 group.")
    
    # Convert figsize list to tuple
    args.figsize = tuple(args.figsize)
    
    # Mode selection
    if args.plot_file:
        # Plot from existing file
        plot_comparison(args.plot_file, args)
    elif args.compare3:
        run_triple_comparison(args)
    elif args.compare:
        run_comparison(args)
    else:
        run_simulation(args)


if __name__ == '__main__':
    main()
