"""
Simple example of Spring-Mass MPC

This is a minimal example showing how to use the SpringMassMPC controller.

Author: MPC Spring-Mass Python Implementation
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from SpringMassMPC import SpringMassMPC


def main():
    print("=" * 60)
    print("Simple Spring-Mass MPC Example")
    print("=" * 60)
    
    # Create a simple 5-mass system
    print("\n1. Creating MPC controller...")
    mpc = SpringMassMPC(
        M=5,           # 5 masses
        m=1.0,         # 1 kg each
        k=5.0,         # Spring stiffness
        c=0.1,         # Friction
        u_max=5.0,     # Max control force
        dt=0.1,        # 100ms timestep
        N=30,          # 30-step prediction horizon (3 seconds)
        Q=1.0,         # Control weight
        R=50.0,        # State weight
        is_linear=True # Linear springs
    )
    print("   ✓ Controller created")
    
    # Set initial conditions (displaced masses)
    print("\n2. Setting initial conditions...")
    x0 = np.array([2.0, -1.5, 1.0, -0.5, 0.8])
    v0 = np.zeros(5)
    print(f"   Initial positions: {x0}")
    print(f"   Initial velocities: {v0}")
    
    # Solve one MPC problem
    print("\n3. Solving MPC optimization problem...")
    x_pred, v_pred, u_pred, success = mpc.solve_mpc(x0, v0, verbose=False)
    
    if success:
        print("   ✓ Optimization succeeded")
    else:
        print("   ⚠ Optimization completed with warnings")
    
    print(f"   Optimal control inputs (first step): u1={u_pred[0,0]:.3f}, u2={u_pred[0,1]:.3f}")
    print(f"   Predicted positions (first step): {x_pred[0]}")
    
    # Visualize the prediction
    print("\n4. Visualizing prediction...")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot predicted positions
    time_vec = mpc.time[:len(x_pred)]
    for i in range(5):
        axes[0].plot(time_vec, x_pred[:, i], label=f'Mass {i+1}', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Position', fontsize=12)
    axes[0].set_title('MPC Predicted Positions', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot control inputs
    axes[1].plot(time_vec, u_pred[:, 0], 'b-', linewidth=2, label='u1 (first mass)')
    axes[1].plot(time_vec, u_pred[:, 1], 'r-', linewidth=2, label='u2 (last mass)')
    axes[1].axhline(y=mpc.u_max, color='k', linestyle='--', alpha=0.3, label='Bounds')
    axes[1].axhline(y=-mpc.u_max, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Control Force', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_title('MPC Control Inputs', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/beer/NBEL/soft_mpc/simple_example.png', dpi=300, bbox_inches='tight')
    print("   ✓ Plot saved as: simple_example.png")
    
    # Run a few control steps
    print("\n5. Running 10 MPC control steps...")
    x_current = x0.copy()
    v_current = v0.copy()
    
    trajectory = [x_current.copy()]
    controls = []
    
    for step in range(10):
        x_next, v_next, u_applied, _ = mpc.step(x_current, v_current, noise_magnitude=0.0)
        trajectory.append(x_next.copy())
        controls.append(u_applied)
        x_current = x_next
        v_current = v_next
        
        if step % 3 == 0:
            max_pos = np.max(np.abs(x_current))
            print(f"   Step {step:2d}: max position = {max_pos:.4f}")
    
    print("\n6. Results:")
    print(f"   Initial max position: {np.max(np.abs(x0)):.4f}")
    print(f"   Final max position:   {np.max(np.abs(x_current)):.4f}")
    print(f"   Reduction: {(1 - np.max(np.abs(x_current))/np.max(np.abs(x0)))*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()

