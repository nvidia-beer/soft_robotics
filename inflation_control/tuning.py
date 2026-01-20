#!/usr/bin/env python3
"""
PID Tuning for Classic PID Controller (figure_classic.py)

Simple empirical tuning method:
1. Binary search Kp for best steady-state response
2. Ki = Kp/3 (integral to eliminate steady-state error)
3. Kd = Kp/500 (minimal - D doesn't help this oscillatory system)

Usage:
    ./run_tuning.sh              # Run tuning
    ./run_tuning.sh --validate   # With validation

Output:
    pid_gains.txt - PID gains for figure_snn.py and figure_classic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

import warp as wp

from models import BalloonModel
from solvers import SolverImplicitFEM
from controllers.pid import PID


def evaluate_kp(kp, dt, test_duration, target_volume_ratio, device):
    """Evaluate a single Kp value and return the average error."""
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
    
    pid = PID(dt=dt, Kp=kp, Ki=0.0, Kd=0.0, u_max=10.0)
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
    
    pid.reset()
    
    # Run test
    n_steps = int(test_duration / dt)
    target_volume = model.initial_volume * target_volume_ratio
    volumes = []
    
    for _ in range(n_steps):
        current_volume = model.compute_current_volume(state)
        volumes.append(current_volume / model.initial_volume)
        pressure = pid.compute(target_volume, current_volume)
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
    
    # Compute metrics
    volumes = np.array(volumes)
    last_n = int(4.0 / dt)
    avg_vol = np.mean(volumes[-last_n:])
    avg_error = abs(target_volume_ratio - avg_vol)
    
    # Penalize overshoot and oscillation (prefer smoother response)
    overshoot = max(0, np.max(volumes) - target_volume_ratio)
    oscillation = np.std(volumes[-last_n:])  # std in steady state
    
    # Combined score: error + overshoot penalty + oscillation penalty
    score = avg_error + 0.5 * overshoot + 0.5 * oscillation
    
    return score, avg_vol


def find_best_kp(
    dt: float = 0.01,
    test_duration: float = 10.0,
    target_volume_ratio: float = 2.0,
    kp_min: int = 15,
    kp_max: int = 35,
    device: str = 'cuda',
):
    """
    Find best Kp (integer) using binary search.
    
    Returns:
        int: Best Kp value
    """
    wp.init()
    
    print("\n" + "=" * 60)
    print("  Finding Best Kp (Binary Search)")
    print("=" * 60)
    print(f"\n  Search range: [{kp_min}, {kp_max}]")
    print(f"  Target volume ratio: {target_volume_ratio}")
    print()
    
    # Cache evaluated results
    cache = {}
    
    def eval_cached(kp):
        kp = int(kp)
        if kp not in cache:
            score, vol = evaluate_kp(kp, dt, test_duration, target_volume_ratio, device)
            cache[kp] = (score, vol)
            print(f"  Kp={kp:3d}: avg_vol={vol:.3f}, score={score:.4f}")
        return cache[kp]
    
    lo, hi = kp_min, kp_max
    
    while hi - lo > 2:
        mid1 = lo + (hi - lo) // 3
        mid2 = hi - (hi - lo) // 3
        
        err1, _ = eval_cached(mid1)
        err2, _ = eval_cached(mid2)
        
        if err1 < err2:
            hi = mid2
        else:
            lo = mid1
    
    # Evaluate remaining candidates
    best_kp = lo
    best_error = float('inf')
    for kp in range(lo, hi + 1):
        err, _ = eval_cached(kp)
        if err < best_error:
            best_error = err
            best_kp = kp
    
    print(f"\n  Best Kp: {best_kp} (error={best_error:.4f}, {len(cache)} evaluations)")
    return best_kp


def compute_pid_gains(Kp: float):
    """
    Compute Ki and Kd from Kp using simple ratios.
    
    For this balloon system:
    - Ki = Kp / 3   (integral to eliminate steady-state error)
    - Kd = Kp / 500 (minimal - D doesn't help this oscillatory system)
    """
    Ki = Kp / 3.0
    Kd = Kp / 500.0
    return Kp, Ki, Kd


def validate_gains(Kp, Ki, Kd, name, dt=0.01, total_time=10.0, device='cuda'):
    """Validate PID gains with a simulation."""
    wp.init()
    
    print(f"\n  {name}: Kp={Kp:.1f}, Ki={Ki:.1f}, Kd={Kd:.2f}")
    
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
    
    pid = PID(dt=dt, Kp=Kp, Ki=Ki, Kd=Kd, u_max=10.0)
    
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
    
    pid.reset()
    
    # Run
    target_ratio = 2.0
    target_volume = model.initial_volume * target_ratio
    n_steps = int(total_time / dt)
    volumes = []
    
    for _ in range(n_steps):
        current_volume = model.compute_current_volume(state)
        pressure = pid.compute(target_volume, current_volume)
        model.apply_inflation(state, pressure, 1.0)
        solver.step(state, state_next, dt, external_forces=None)
        state, state_next = state_next, state
        volumes.append(current_volume / model.initial_volume)
    
    volumes = np.array(volumes)
    final_vol = volumes[-1]
    final_error = abs(target_ratio - final_vol)
    overshoot = max(0, np.max(volumes) - target_ratio)
    
    print(f"    Final: {final_vol:.3f} (error={final_error:.4f}, overshoot={overshoot:.3f})")
    return final_vol, final_error, overshoot


def main():
    parser = argparse.ArgumentParser(description='PID Tuning')
    parser.add_argument('--kp-min', type=int, default=15)
    parser.add_argument('--kp-max', type=int, default=35)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  PID Tuning")
    print("=" * 60)
    
    # Step 1: Find best Kp using binary search
    best_kp = find_best_kp(
        kp_min=args.kp_min,
        kp_max=args.kp_max,
        device=args.device,
    )
    
    # Step 2: Compute Ki and Kd
    Kp, Ki, Kd = compute_pid_gains(best_kp)
    
    print("\n" + "=" * 60)
    print("  Computed PID Gains")
    print("=" * 60)
    print(f"\n  Kp = {Kp}  (from search)")
    print(f"  Ki = {Ki:.1f}  (= Kp / 3)")
    print(f"  Kd = {Kd:.3f}  (= Kp / 500, minimal)")
    
    # Step 3: Validate
    if args.validate:
        print("\n" + "=" * 60)
        print("  Validation")
        print("=" * 60)
        
        validate_gains(Kp, 0.0, 0.0, "P", device=args.device)
        validate_gains(Kp, Ki, 0.0, "PI", device=args.device)
    
    # Step 4: Save to file
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pid_gains.txt')
    with open(config_file, 'w') as f:
        f.write(f"# PID gains from tuning.py\n")
        f.write(f"KP={Kp}\n")
        f.write(f"KI={Ki:.1f}\n")
        f.write(f"KD={Kd:.3f}\n")
    
    print("\n" + "=" * 60)
    print(f"  Saved to: {config_file}")
    print("=" * 60)
    print(f"""
KP = {Kp}
KI = {Ki:.1f}
KD = {Kd:.3f}
""")
    
    print("=" * 60)
    print("  Done! Run ./create_figure.sh 2 to generate figure")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
