#!/usr/bin/env python3
"""
SNN PID Tuning for Neuromorphic Controller (figure_snn.py)

Searches for optimal Kp for the SNN controller, which needs MUCH lower
gains than classic PID due to neural dynamics (synaptic delays).

Usage:
    ./run_snn_tuning.sh              # Run tuning
    ./run_snn_tuning.sh --validate   # With validation

Output:
    snn_gains.txt - SNN PID gains for figure_snn.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import nengo

import warp as wp

from models import BalloonModel
from solvers import SolverImplicitFEM
from controllers.nengo import SNN_PID_Controller


def evaluate_snn_kp(kp, dt, test_duration, target_volume_ratio, device, n_neurons=500):
    """Evaluate a single Kp value for SNN and return the average error."""
    
    model = BalloonModel(
        radius=0.5, num_boundary=16, num_rings=2,
        max_volume_ratio=target_volume_ratio + 0.5, device=device, boxsize=3.0,
        spring_stiffness=1000.0, spring_damping=5.0,
        fem_E=2000.0, fem_nu=0.45, fem_damping=10.0,
    )
    
    solver = SolverImplicitFEM(
        model, dt=dt, mass=1.0, preconditioner_type="diag",
        solver_type="bicgstab", max_iterations=50, tolerance=1e-4,
        rebuild_matrix_every=1,
    )
    
    state = model.state()
    state_next = model.state()
    
    # Pre-inflate to ratio 1.0
    for _ in range(50):
        current_vol = model.compute_current_volume(state)
        if abs(current_vol - model.initial_volume) < 0.01:
            break
        error = model.initial_volume - current_vol
        pressure = np.clip(error * 5.0, -10, 10)
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    # Build SNN controller (P-only for Kp search)
    current_error = [0.0]
    snn_output = [0.0]
    last_pressure = [0.0]  # For rate limiting
    MAX_PRESSURE_RATE = 1.0  # Match figure_snn.py
    error_scale = 1.0
    u_max = 10.0
    
    def get_error(t):
        return current_error[0] / error_scale
    
    nengo_model = nengo.Network(label="SNN_Tuning")
    with nengo_model:
        controller = SNN_PID_Controller(
            Kp=kp,
            Ki=0.0,  # P-only for Kp search
            Kd=0.0,
            u_max=u_max,
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
    
    nengo_dt = 0.001
    sim = nengo.Simulator(nengo_model, dt=nengo_dt)
    physics_steps_per_nengo = max(1, int(dt / nengo_dt))
    
    # Run test
    n_steps = int(test_duration / dt)
    target_volume = model.initial_volume * target_volume_ratio
    volumes = []
    
    for _ in range(n_steps):
        current_volume = model.compute_current_volume(state)
        current_ratio = current_volume / model.initial_volume
        volumes.append(current_ratio)
        
        error_ratio = target_volume_ratio - current_ratio
        current_error[0] = error_ratio
        
        # Run Nengo steps
        for _ in range(physics_steps_per_nengo):
            sim.step()
        
        # Rate limiting (same as figure_snn.py)
        raw_pressure = snn_output[0]
        delta = raw_pressure - last_pressure[0]
        delta = np.clip(delta, -MAX_PRESSURE_RATE, MAX_PRESSURE_RATE)
        pressure = last_pressure[0] + delta
        last_pressure[0] = pressure
        
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    sim.close()
    
    # Compute metrics
    volumes = np.array(volumes)
    last_n = int(4.0 / dt)
    avg_vol = np.mean(volumes[-last_n:])
    avg_error = abs(target_volume_ratio - avg_vol)
    
    # Penalize overshoot and oscillation
    overshoot = max(0, np.max(volumes) - target_volume_ratio)
    oscillation = np.std(volumes[-last_n:])
    
    # Also penalize early oscillation (first half of simulation)
    early_osc = np.std(volumes[:len(volumes)//2])
    
    # Combined score - heavily penalize oscillation for stability
    score = avg_error + 1.0 * overshoot + 2.0 * oscillation + 1.0 * early_osc
    
    return score, avg_vol, overshoot, oscillation


def find_best_snn_kp(
    dt: float = 0.01,
    test_duration: float = 10.0,  # Match figure_snn.py
    target_volume_ratio: float = 2.0,
    kp_min: float = 1.5,
    kp_max: float = 5.0,
    kp_step: float = 0.5,
    device: str = 'cuda',
    n_neurons: int = 500,  # Match figure_snn.py
):
    """
    Find best Kp for SNN using linear search.
    
    SNN needs Kp_norm in [0, 1] range.
    With error_scale=1.0, u_max=10: Kp_norm = Kp / 10
    So Kp range [1.0, 8.0] gives Kp_norm range [0.1, 0.8]
    """
    wp.init()
    
    print("\n" + "=" * 60)
    print("  Finding Best SNN Kp (Linear Search)")
    print("=" * 60)
    print(f"\n  Search range: [{kp_min}, {kp_max}], step={kp_step}")
    print(f"  Target volume ratio: {target_volume_ratio}")
    print(f"  Neurons: {n_neurons}")
    print()
    
    results = []
    kp = kp_min
    while kp <= kp_max:
        score, vol, overshoot, osc = evaluate_snn_kp(
            kp, dt, test_duration, target_volume_ratio, device, n_neurons
        )
        error = abs(target_volume_ratio - vol)
        results.append((kp, score, vol, overshoot, osc, error))
        print(f"  Kp={kp:.1f}: vol={vol:.3f}, error={error:.3f}, overshoot={overshoot:.3f}, osc={osc:.3f}, score={score:.4f}")
        kp += kp_step
    
    # Find best
    best = min(results, key=lambda x: x[1])
    best_kp = best[0]
    
    print(f"\n  Best Kp: {best_kp} (error={best[5]:.3f}, score={best[1]:.4f})")
    return best_kp


def compute_snn_gains(Kp: float):
    """
    Compute Ki and Kd from Kp for SNN.
    
    SNN ratios (tuned for neural dynamics with rate limiting):
    - Ki = Kp * 0.5  (integral for steady-state accuracy)
    - Kd = Kp * 0.02 (minimal derivative - D term destabilizes SNN)
    """
    Ki = Kp * 0.5
    Kd = Kp * 0.02  # Very small - D term causes oscillation in SNN
    return Kp, Ki, Kd


def validate_snn_gains(Kp, Ki, Kd, name, dt=0.01, total_time=10.0, device='cuda', n_neurons=500):
    """Validate SNN gains with a simulation."""
    wp.init()
    
    print(f"\n  {name}: Kp={Kp:.1f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
    
    model = BalloonModel(
        radius=0.5, num_boundary=16, num_rings=2,
        max_volume_ratio=2.5, device=device, boxsize=3.0,
        spring_stiffness=1000.0, spring_damping=5.0,
        fem_E=2000.0, fem_nu=0.45, fem_damping=10.0,
    )
    
    solver = SolverImplicitFEM(
        model, dt=dt, mass=1.0, preconditioner_type="diag",
        solver_type="bicgstab", max_iterations=50, tolerance=1e-4,
    )
    
    state = model.state()
    state_next = model.state()
    
    # Pre-inflate
    for _ in range(50):
        current_vol = model.compute_current_volume(state)
        if abs(current_vol - model.initial_volume) < 0.01:
            break
        error = model.initial_volume - current_vol
        pressure = np.clip(error * 5.0, -10, 10)
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    # Build SNN
    current_error = [0.0]
    snn_output = [0.0]
    last_pressure = [0.0]  # For rate limiting
    MAX_PRESSURE_RATE = 1.0  # Match figure_snn.py
    error_scale = 1.0
    u_max = 10.0
    
    def get_error(t):
        return current_error[0] / error_scale
    
    nengo_model = nengo.Network(label=f"SNN_{name}")
    with nengo_model:
        controller = SNN_PID_Controller(
            Kp=Kp, Ki=Ki, Kd=Kd,
            u_max=u_max,
            n_neurons=n_neurons,
            error_scale=error_scale,
        )
        controller.build(model=nengo_model, get_error_callback=get_error)
        
        def sync_output(t):
            snn_output[0] = controller.get_output()
        nengo.Node(sync_output, size_in=0, size_out=0)
    
    nengo_dt = 0.001
    sim = nengo.Simulator(nengo_model, dt=nengo_dt)
    physics_steps_per_nengo = max(1, int(dt / nengo_dt))
    
    # Run
    target_ratio = 2.0
    n_steps = int(total_time / dt)
    volumes = []
    
    for _ in range(n_steps):
        current_volume = model.compute_current_volume(state)
        current_ratio = current_volume / model.initial_volume
        volumes.append(current_ratio)
        
        current_error[0] = target_ratio - current_ratio
        
        for _ in range(physics_steps_per_nengo):
            sim.step()
        
        # Rate limiting (same as figure_snn.py)
        raw_pressure = snn_output[0]
        delta = raw_pressure - last_pressure[0]
        delta = np.clip(delta, -MAX_PRESSURE_RATE, MAX_PRESSURE_RATE)
        pressure = last_pressure[0] + delta
        last_pressure[0] = pressure
        
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    sim.close()
    
    volumes = np.array(volumes)
    final_vol = volumes[-1]
    final_error = abs(target_ratio - final_vol)
    overshoot = max(0, np.max(volumes) - target_ratio)
    oscillation = np.std(volumes[-int(4.0/dt):])
    
    print(f"    Final: {final_vol:.3f} (error={final_error:.4f}, overshoot={overshoot:.3f}, osc={oscillation:.3f})")
    return final_vol, final_error, overshoot


def main():
    parser = argparse.ArgumentParser(description='SNN PID Tuning')
    parser.add_argument('--kp-min', type=float, default=1.5)
    parser.add_argument('--kp-max', type=float, default=5.0)
    parser.add_argument('--kp-step', type=float, default=0.5)
    parser.add_argument('--neurons', type=int, default=500)  # Match figure_snn.py
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  SNN PID Tuning")
    print("=" * 60)
    
    # Step 1: Find best Kp
    best_kp = find_best_snn_kp(
        kp_min=args.kp_min,
        kp_max=args.kp_max,
        kp_step=args.kp_step,
        device=args.device,
        n_neurons=args.neurons,
    )
    
    # Step 2: Compute Ki and Kd
    Kp, Ki, Kd = compute_snn_gains(best_kp)
    
    print("\n" + "=" * 60)
    print("  Computed SNN Gains")
    print("=" * 60)
    print(f"\n  Kp = {Kp:.1f}  (from search)")
    print(f"  Ki = {Ki:.2f}  (= Kp * 0.5)")
    print(f"  Kd = {Kd:.2f}  (= Kp * 0.02, minimal to avoid D-term oscillation)")
    
    # Step 3: Validate
    if args.validate:
        print("\n" + "=" * 60)
        print("  Validation")
        print("=" * 60)
        
        validate_snn_gains(Kp, 0.0, 0.0, "P", device=args.device, n_neurons=args.neurons)
        validate_snn_gains(Kp, Ki, 0.0, "PI", device=args.device, n_neurons=args.neurons)
        validate_snn_gains(Kp, 0.0, Kd, "PD", device=args.device, n_neurons=args.neurons)
        validate_snn_gains(Kp, Ki, Kd, "PID", device=args.device, n_neurons=args.neurons)
    
    # Step 4: Save to file
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snn_gains.txt')
    with open(config_file, 'w') as f:
        f.write(f"# SNN PID gains from snn_tuning.py\n")
        f.write(f"KP={Kp:.1f}\n")
        f.write(f"KI={Ki:.2f}\n")
        f.write(f"KD={Kd:.2f}\n")
    
    print("\n" + "=" * 60)
    print(f"  Saved to: {config_file}")
    print("=" * 60)
    print(f"""
KP = {Kp:.1f}
KI = {Ki:.2f}
KD = {Kd:.2f}
""")
    
    print("=" * 60)
    print("  Done! Run ./create_figure.sh 1 to generate figure")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
