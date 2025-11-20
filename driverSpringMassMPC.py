"""
Driver code for Spring-Mass MPC Controller

This script demonstrates the use of the SpringMassMPC controller
on a system of masses connected by springs.

Author: MPC Spring-Mass Python Implementation
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from SpringMassMPC import SpringMassMPC


def main():
    """
    Main function to run MPC simulation
    """
    
    ###########################################################################
    # System Parameters
    ###########################################################################
    M = 7           # Number of masses
    m = 1.0         # Mass of each element
    k = 5.0         # Linear spring coefficient
    c = 0.1         # Friction coefficient
    k_nl = 0.01     # Nonlinear spring coefficient
    u_max = 5.0     # Maximum control force
    
    ###########################################################################
    # MPC Parameters
    ###########################################################################
    dt = 0.1        # Time step
    N = 50          # Prediction horizon (reduced for faster computation)
    Q = 1.0         # Control weight
    R = 50.0        # State weight
    is_linear = True  # Use linear spring model for MPC
    
    ###########################################################################
    # Create MPC Controller
    ###########################################################################
    print("Creating MPC controller...")
    mpc = SpringMassMPC(M=M, m=m, k=k, c=c, k_nl=k_nl, 
                        u_max=u_max, dt=dt, N=N, Q=Q, R=R,
                        is_linear=is_linear)
    
    ###########################################################################
    # Initial Conditions
    ###########################################################################
    x0 = np.array([-1.0, 3.0, 1.5, -4.0, 0.3, -0.5, -0.3])[:M]
    v0 = np.zeros(M)
    
    ###########################################################################
    # Simulate Uncontrolled System
    ###########################################################################
    print("\nSimulating uncontrolled system...")
    T_sim = 100.0  # Total simulation time
    t_uncontrolled, x_uncontrolled = mpc.simulate_uncontrolled(x0, v0, T_sim)
    
    ###########################################################################
    # Simulate Controlled System with MPC
    ###########################################################################
    print("\nSimulating controlled system with MPC...")
    
    # Reset for controlled simulation
    mpc_controlled = SpringMassMPC(M=M, m=m, k=k, c=c, k_nl=k_nl,
                                   u_max=u_max, dt=dt, N=N, Q=Q, R=R,
                                   is_linear=is_linear)
    
    # Simulation parameters
    n_steps = 500
    noise_magnitude = 0.2 * 2 * u_max  # Noise magnitude
    
    # Initial state
    x_current = x0.copy()
    v_current = v0.copy()
    
    # Storage for controlled trajectory
    t_controlled = []
    x_controlled = []
    u_controlled = []
    x_predictions = []
    
    # Run MPC loop
    for i in range(n_steps):
        if i % 10 == 0:
            print(f"  Step {i}/{n_steps}")
        
        try:
            x_next, v_next, u_applied, x_pred = mpc_controlled.step(
                x_current, v_current, noise_magnitude
            )
            
            t_controlled.append(mpc_controlled.current_time)
            x_controlled.append(x_current.copy())
            u_controlled.append(u_applied)
            x_predictions.append(x_pred)
            
            x_current = x_next
            v_current = v_next
            
        except Exception as e:
            print(f"  Error at step {i}: {e}")
            break
    
    print(f"\nCompleted {len(t_controlled)} steps")
    
    # Convert to arrays
    t_controlled = np.array(t_controlled)
    x_controlled = np.array(x_controlled)
    u_controlled = np.array(u_controlled)
    
    ###########################################################################
    # Plot Results
    ###########################################################################
    print("\nGenerating plots...")
    
    # Plot 1: Uncontrolled vs Controlled System
    fig1, axes = plt.subplots(M, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle('Spring-Mass System: Uncontrolled vs Controlled', fontsize=14, fontweight='bold')
    
    for i in range(M):
        ax = axes[i] if M > 1 else axes
        
        # Uncontrolled
        ax.plot(t_uncontrolled, x_uncontrolled[:, i], 
               'b-', alpha=0.3, linewidth=2, label='Uncontrolled')
        
        # Controlled
        if len(x_controlled) > 0:
            ax.plot(t_controlled, x_controlled[:, i], 
                   'r-', alpha=0.7, linewidth=2, label='Controlled')
        
        ax.set_ylabel(f'$x_{i+1}$', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')
        if i == M-1:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    plt.xlim([0, 50])
    plt.tight_layout()
    plt.savefig('/home/beer/NBEL/soft_mpc/controlled_vs_uncontrolled.png', dpi=300, bbox_inches='tight')
    print("  Saved: controlled_vs_uncontrolled.png")
    
    # Plot 2: Control Inputs
    if len(u_controlled) > 0:
        fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig2.suptitle('Control Inputs', fontsize=14, fontweight='bold')
        
        axes2[0].plot(t_controlled, [u[0] for u in u_controlled], 'b-', linewidth=2)
        axes2[0].set_ylabel('$u_1$ (Force on first mass)', fontsize=10)
        axes2[0].grid(True, alpha=0.3)
        axes2[0].axhline(y=u_max, color='r', linestyle='--', label=f'Max = ±{u_max}')
        axes2[0].axhline(y=-u_max, color='r', linestyle='--')
        axes2[0].legend()
        
        axes2[1].plot(t_controlled, [u[1] for u in u_controlled], 'g-', linewidth=2)
        axes2[1].set_ylabel('$u_2$ (Force on last mass)', fontsize=10)
        axes2[1].set_xlabel('Time (s)', fontsize=10)
        axes2[1].grid(True, alpha=0.3)
        axes2[1].axhline(y=u_max, color='r', linestyle='--', label=f'Max = ±{u_max}')
        axes2[1].axhline(y=-u_max, color='r', linestyle='--')
        axes2[1].legend()
        
        plt.tight_layout()
        plt.savefig('/home/beer/NBEL/soft_mpc/control_inputs.png', dpi=300, bbox_inches='tight')
        print("  Saved: control_inputs.png")
    
    # Plot 3: Controlled trajectory with predictions
    fig3, axes3 = plt.subplots(M, 1, figsize=(12, 10), sharex=True)
    fig3.suptitle('Controlled System with MPC Predictions', fontsize=14, fontweight='bold')
    
    for i in range(M):
        ax = axes3[i] if M > 1 else axes3
        
        # Controlled trajectory
        if len(x_controlled) > 0:
            ax.plot(t_controlled, x_controlled[:, i], 
                   'r-', alpha=0.7, linewidth=2, label='Actual')
        
        # MPC predictions (plot every 10th prediction to avoid clutter)
        if len(x_predictions) > 0:
            for j in range(0, len(x_predictions), 10):
                t_pred = t_controlled[j] + mpc_controlled.time[:len(x_predictions[j])]
                ax.plot(t_pred, x_predictions[j][:, i], 
                       'b-', alpha=0.1, linewidth=0.5)
        
        ax.set_ylabel(f'$x_{i+1}$', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylim([-1, 1])
        
        if i == 0:
            ax.legend(loc='upper right')
        if i == M-1:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    plt.xlim([0, 30])
    plt.tight_layout()
    plt.savefig('/home/beer/NBEL/soft_mpc/controlled_with_predictions.png', dpi=300, bbox_inches='tight')
    print("  Saved: controlled_with_predictions.png")
    
    plt.show()
    
    print("\n=== Simulation Complete ===")
    print(f"Final positions: {x_current}")
    print(f"Final velocities: {v_current}")
    print(f"Max position deviation: {np.max(np.abs(x_current)):.4f}")


if __name__ == "__main__":
    main()

