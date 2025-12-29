#!/usr/bin/env python3
"""
Simple SNN CPG Demo - Spiking Neural Network without PES

Same as demo_simple_cpg.py but uses Nengo spiking neurons instead of rate-coded CPG.
No PES learning, no GUI - just pure spiking oscillators + physics.

Usage:
    python demo_snn_simple.py
    
    Or via run_snn.sh

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))

import numpy as np
import warp as wp
import pygame
import argparse
import time
import nengo

from spring_mass_env import SpringMassEnv
from sim import Model
from inject_forces_locomotion import InjectForcesLocomotion, get_available_modes, describe_mode


def hopf_dynamics(x, a=5.0, mu=1.0, omega=2*np.pi*2.0, tau=0.01):
    """
    Hopf oscillator dynamics in Cartesian coordinates.
    
    NEF recurrent: x_new ≈ synapse(f(x))
    For integration: f(x) = x + tau * derivative
    This implements: dx/dt = derivative via the lowpass synapse
    """
    r_sq = x[0]**2 + x[1]**2
    dx = a * (mu - r_sq) * x[0] - omega * x[1]
    dy = a * (mu - r_sq) * x[1] + omega * x[0]
    # CRITICAL: Return x + tau*dx for proper NEF integration!
    # The synapse implements: y → (1-dt/tau)*y + (dt/tau)*f(y)
    # With f(x) = x + tau*derivative, this gives integration
    return [x[0] + tau * dx, x[1] + tau * dy]


class SimpleSNNCPG:
    """
    Simple Spiking Neural Network CPG using Nengo.
    
    Each group has a Hopf oscillator implemented as a 2D spiking ensemble.
    No PES learning - just pure oscillator dynamics.
    """
    
    def __init__(
        self,
        num_groups: int,
        frequency: float = 4.0,
        amplitude: float = 1.0,
        direction: tuple = (1.0, 0.0),
        n_neurons: int = 100,
        dt: float = 0.001,
    ):
        self.num_groups = num_groups
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = np.array(direction)
        self.n_neurons = n_neurons
        self.dt = dt
        
        self.grid_side = int(np.sqrt(num_groups))
        self.omega = 2 * np.pi * frequency
        
        # Build Nengo network
        self._build_network()
        
        # Create simulator
        self.sim = nengo.Simulator(self.model, dt=dt)
        
        print(f"[SimpleSNNCPG] Initialized:")
        print(f"  Groups: {num_groups} ({self.grid_side}x{self.grid_side})")
        print(f"  Frequency: {frequency} Hz")
        print(f"  Neurons per oscillator: {n_neurons}")
        print(f"  Direction: [{self.direction[0]:.2f}, {self.direction[1]:.2f}]")
    
    def _build_network(self):
        """Build Nengo network with Hopf oscillators."""
        self.model = nengo.Network(label="SimpleSNNCPG")
        
        # Phase offsets for traveling wave (same as classic HopfCPG)
        phase_per_cell = np.pi / 2.0  # 90° per cell = proper traveling wave
        tau_syn = 0.01
        
        # Compute phase offsets
        self.phase_offsets = np.zeros(self.num_groups)
        for i in range(self.num_groups):
            row = i // self.grid_side
            col = i % self.grid_side
            self.phase_offsets[i] = (col * self.direction[0] + row * self.direction[1]) * phase_per_cell
        
        with self.model:
            # Create oscillator ensembles
            self.oscillators = []
            self.output_probes = []
            
            for i in range(self.num_groups):
                row = i // self.grid_side
                col = i % self.grid_side
                phase = self.phase_offsets[i]
                
                # Create 2D ensemble for Hopf oscillator
                osc = nengo.Ensemble(
                    n_neurons=self.n_neurons,
                    dimensions=2,
                    radius=1.2,
                    label=f"osc_{i}",
                )
                
                # Initialize on limit cycle at correct phase
                initial_x = np.cos(phase)
                initial_y = np.sin(phase)
                
                def make_init(x0, y0):
                    def init_fn(t):
                        if t < 0.5:  # Initialize for 0.5s
                            return [x0 * 5, y0 * 5]
                        return [0, 0]
                    return init_fn
                
                init_node = nengo.Node(make_init(initial_x, initial_y), size_out=2)
                nengo.Connection(init_node, osc, synapse=0.01)
                
                # Hopf dynamics - this is all we need!
                def make_hopf(omega=self.omega, tau=tau_syn):
                    def dynamics(x):
                        return hopf_dynamics(x, a=10.0, mu=1.0, omega=omega, tau=tau)
                    return dynamics
                
                nengo.Connection(
                    osc, osc,
                    function=make_hopf(self.omega, tau_syn),
                    synapse=tau_syn,
                )
                
                # Output probe (x component = force)
                probe = nengo.Probe(osc, synapse=0.01)
                
                self.oscillators.append(osc)
                self.output_probes.append(probe)
            
            # No inter-oscillator coupling needed!
            # All oscillators run at same ω, phases set during initialization
            print(f"  Created {self.num_groups} Hopf oscillators (no coupling)")
    
    def step(self):
        """Run one simulation step and return outputs."""
        self.sim.step()
        
        # Get outputs (x component of each oscillator)
        outputs = np.zeros(self.num_groups)
        for i, probe in enumerate(self.output_probes):
            # Get latest value from probe
            if len(self.sim.data[probe]) > 0:
                state = self.sim.data[probe][-1]
                outputs[i] = state[0] * self.amplitude  # x component
        
        return np.clip(outputs, -1.0, 1.0)
    
    def reset(self):
        """Reset simulator."""
        self.sim.reset()
    
    def close(self):
        """Close simulator."""
        self.sim.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Simple SNN CPG Demo')
    
    # Grid configuration
    parser.add_argument('--grid-size', '-n', type=int, default=4,
                        help='Grid size NxN (default: 4)')
    
    # Physics
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device (default: cuda)')
    
    # CPG parameters
    parser.add_argument('--frequency', type=float, default=4.0,
                        help='CPG frequency in Hz (default: 4.0, matches classic)')
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='CPG amplitude 0-1 (default: 1.0, matches classic)')
    parser.add_argument('--direction', type=float, nargs=2, default=[1.0, 0.0],
                        metavar=('DX', 'DY'),
                        help='Movement direction 2D vector (default: 1 0 = right)')
    parser.add_argument('--n-neurons', type=int, default=100,
                        help='Neurons per oscillator (default: 100)')
    
    # Force parameters
    parser.add_argument('--force-scale', type=float, default=20.0,
                        help='Force scale multiplier (default: 20.0, matches classic)')
    
    # Display
    parser.add_argument('--window-width', type=int, default=1000,
                        help='Window width (default: 1000)')
    parser.add_argument('--window-height', type=int, default=500,
                        help='Window height (default: 500)')
    parser.add_argument('--boxsize', type=float, default=2.5,
                        help='Simulation box height (default: 2.5)')
    
    # Duration
    parser.add_argument('--duration', '-t', type=float, default=30.0,
                        help='Simulation duration in seconds (default: 30)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SIMPLE SNN CPG DEMO - Spiking Neural Network (No PES)")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Create Environment
    # =========================================================================
    print("Creating environment...")
    
    env = SpringMassEnv(
        render_mode='human',
        N=args.grid_size,
        dt=args.dt,
        spring_coeff=50.0,
        spring_damping=0.3,
        gravity=-0.5,
        boxsize=args.boxsize,
        device=args.device,
        with_fem=True,
        with_springs=True,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    
    # Replace solver with ratchet friction
    direction = tuple(args.direction)
    print(f"Setting up solver with ratchet friction, direction = {direction}...")
    from solvers import SolverImplicit
    
    env.solver = SolverImplicit(
        env.model,
        dt=args.dt,
        mass=1.0,
        preconditioner_type="diag",
        solver_type="bicgstab",
        max_iterations=30,
        tolerance=1e-3,
        ratchet_friction=True,
        locomotion_direction=direction,
    )
    
    env.reset(seed=42)
    
    # Lower body to ground
    initial_pos = env.model.particle_q.numpy()
    min_y_initial = np.min(initial_pos[:, 1])
    ground_margin = 0.02
    y_offset = min_y_initial - ground_margin
    if y_offset > 0:
        print(f"  Lowering body by {y_offset:.3f} to touch ground...")
        initial_pos[:, 1] -= y_offset
        new_pos_wp = wp.array(initial_pos, dtype=wp.vec2, device=env.state_in.particle_q.device)
        env.state_in.particle_q = new_pos_wp
        env.state_out.particle_q = wp.clone(new_pos_wp)
    
    print(f"  Grid: {args.grid_size}x{args.grid_size} = {args.grid_size**2} particles")
    print(f"  Device: {args.device}")
    print()
    
    # =========================================================================
    # Create SNN CPG
    # =========================================================================
    num_groups = (args.grid_size - 1) ** 2
    
    cpg = SimpleSNNCPG(
        num_groups=num_groups,
        frequency=args.frequency,
        amplitude=args.amplitude,
        direction=args.direction,
        n_neurons=args.n_neurons,
        dt=args.dt,
    )
    
    print(f"\nSNN CPG Configuration:")
    print(f"  Type: Spiking Hopf Oscillator (Nengo)")
    print(f"  Groups: {num_groups}")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Neurons: {args.n_neurons} per oscillator")
    print()
    
    # =========================================================================
    # Create Force Injector (horizontal like classic demo)
    # =========================================================================
    injector = InjectForcesLocomotion(
        env.model,
        group_size=2,
        device=args.device,
        mode='horizontal',  # Match classic demo (not radial which pushes up)
        force_scale=args.force_scale,
    )
    
    print(f"Force Configuration:")
    print(f"  Mode: HORIZONTAL - Like classic demo")
    print(f"  Scale: {args.force_scale}")
    print()
    
    # =========================================================================
    # Initialize
    # =========================================================================
    pygame.init()
    
    initial_positions = env.model.particle_q.numpy()
    initial_centroid_x = np.mean(initial_positions[:, 0])
    min_y_initial = np.min(initial_positions[:, 1])
    GROUND_CONTACT_THRESHOLD = min_y_initial + 0.1
    
    print("=" * 70)
    print("STARTING SIMULATION")
    print("=" * 70)
    print()
    print("Press Q or ESC to quit")
    print("Press R to reset")
    print("Press SPACE to pause/resume")
    print()
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    t = 0.0
    paused = False
    running = True
    frame_count = 0
    start_time = time.time()
    
    while running and t < args.duration:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    env.reset(seed=42)
                    cpg.reset()
                    t = 0.0
                    print("Reset!")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        if paused:
            pygame.time.wait(50)
            continue
        
        # Get current positions
        current_positions = env.state_in.particle_q.numpy()
        injector.calculate_centroids(current_positions)
        
        # Get SNN CPG output
        cpg_output = cpg.step()
        
        # Clear and apply forces (only after initialization period)
        # During first 0.5s, oscillators are being kick-started onto limit cycle
        INIT_PERIOD = 0.5
        injector.reset()
        if t > INIT_PERIOD:
            for group_id in range(num_groups):
                cpg_val = cpg_output[group_id]
                if abs(cpg_val) > 0.001:
                    injector.inject_locomotion_force(group_id, cpg_val)
        
        # Step physics
        forces = injector.get_forces_array()
        action = forces.flatten().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Debug: print CPG matrix every second
        if frame_count % 100 == 1:
            max_force = np.max(np.abs(forces))
            total_fx_actual = np.sum(forces[:, 0])
            total_fy_actual = np.sum(forces[:, 1])
            print(f"\n[SNN SIMPLE] t={t:.2f}s")
            print(f"  Forces: max={max_force:.4f}, sum_fx={total_fx_actual:.4f}, sum_fy={total_fy_actual:.4f}")
            # Print CPG matrix
            grid_side = int(np.sqrt(num_groups))
            print(f"  CPG Matrix ({grid_side}x{grid_side}):")
            for row in range(grid_side - 1, -1, -1):  # Top to bottom
                row_vals = []
                for col in range(grid_side):
                    gid = row * grid_side + col
                    row_vals.append(f"{cpg_output[gid]:+.2f}")
                print(f"    [{' '.join(row_vals)}]")
        
        # Calculate displacement
        new_positions = env.state_in.particle_q.numpy()
        current_centroid_x = np.mean(new_positions[:, 0])
        displacement = current_centroid_x - initial_centroid_x
        velocity = displacement / max(t, 0.01)
        
        # Render
        injector.calculate_centroids(new_positions)
        env._sync_to_cpu()
        env._init_window()
        
        if env.window is not None:
            canvas = env._create_canvas()
            scale = env.window_height / env.boxsize
            env._draw_grid(canvas)
            env._draw_fem_triangles(canvas, scale)
            env._draw_springs(canvas, scale)
            env._draw_particles(canvas, scale)
            env._draw_centroids(canvas, scale)
            env._draw_ui_text(canvas)
            env._draw_legends(canvas)
            
            env.window.blit(canvas, canvas.get_rect())
            
            # Draw SNN indicator
            font = pygame.font.Font(None, 24)
            snn_text = font.render("SNN CPG (Spiking)", True, (0, 150, 255))
            env.window.blit(snn_text, (10, env.window_height - 30))
            
            pygame.display.flip()
        
        # Progress
        if frame_count % 100 == 0:
            fps = frame_count / max(time.time() - start_time, 0.01)
            print(f"t={t:.2f}s | disp={displacement:+.4f}m | vel={velocity:+.4f}m/s | fps={fps:.1f}")
        
        t += args.dt
        frame_count += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    
    final_positions = env.state_in.particle_q.numpy()
    final_centroid_x = np.mean(final_positions[:, 0])
    total_displacement = final_centroid_x - initial_centroid_x
    avg_velocity = total_displacement / t if t > 0 else 0
    
    print(f"  Duration: {t:.2f}s")
    print(f"  Total displacement: {total_displacement:+.4f} m")
    print(f"  Average velocity: {avg_velocity:+.4f} m/s")
    print()
    
    if total_displacement > 0.01:
        print("  ✓ Body moved RIGHT - SNN Locomotion working!")
    elif total_displacement < -0.01:
        print("  ← Body moved LEFT")
    else:
        print("  ✗ Body didn't move significantly")
    
    cpg.close()
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

