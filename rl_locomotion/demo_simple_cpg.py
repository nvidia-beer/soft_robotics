#!/usr/bin/env python3
"""
Simple CPG Demo - Validates force application without SNN complexity

This script runs the basic Hopf CPG with the locomotion force injector
to test that forces and ratchet friction work correctly.

No Nengo, no SNN - just pure Python CPG + physics.

Usage:
    python demo_simple_cpg.py
    
    Or via run_simple_cpg.sh

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp
import pygame
import argparse
import time

from spring_mass_env import SpringMassEnv
from sim import Model
from cpg import HopfCPG
from balloon_forces import BalloonForces
from pygame_renderer import Renderer


def parse_args():
    parser = argparse.ArgumentParser(description='Simple CPG Demo - No SNN')
    
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
                        help='CPG frequency in Hz (default: 4.0)')
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='CPG amplitude 0-1 (default: 1.0)')
    parser.add_argument('--direction', type=float, nargs=2, default=[1.0, 0.0],
                        metavar=('DX', 'DY'),
                        help='Movement direction 2D vector (default: 1 0 = right)')
    
    # Force
    parser.add_argument('--force-scale', type=float, default=20.0,
                        help='Force scale multiplier (default: 20.0)')
    
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
    
    # Ground check option
    # Note: Forces are always applied (radial forces create internal deformation)
    # Ground friction converts deformation into locomotion
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SIMPLE CPG DEMO - Force Validation (No SNN)")
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
        spring_coeff=50.0,      # Moderate stiffness - solid but deformable
        spring_damping=0.3,     # Moderate damping
        gravity=-0.5,           # Moderate gravity
        boxsize=args.boxsize,
        device=args.device,
        with_fem=True,
        with_springs=True,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    
    # Replace solver with direction-aware ratchet friction
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
        ratchet_friction=True,  # Enable ratchet friction for locomotion
        locomotion_direction=direction,  # Friction aligned with CPG direction
    )
    
    env.reset(seed=42)
    
    # Check initial position and LOWER body to ground
    initial_pos = env.model.particle_q.numpy()
    min_y_initial = np.min(initial_pos[:, 1])
    max_y_initial = np.max(initial_pos[:, 1])
    print(f"  Initial Y range: [{min_y_initial:.3f}, {max_y_initial:.3f}]")
    
    # Lower the body to touch ground (y=0) with small margin
    ground_margin = 0.02  # Small gap above ground
    y_offset = min_y_initial - ground_margin
    if y_offset > 0:
        print(f"  Lowering body by {y_offset:.3f} to touch ground...")
        initial_pos[:, 1] -= y_offset
        
        # Update solver states (solver uses state_in/state_out, not model!)
        new_pos_wp = wp.array(initial_pos, dtype=wp.vec2, device=env.state_in.particle_q.device)
        env.state_in.particle_q = new_pos_wp
        env.state_out.particle_q = wp.clone(new_pos_wp)
        
        min_y_initial = np.min(initial_pos[:, 1])
        max_y_initial = np.max(initial_pos[:, 1])
        print(f"  New Y range: [{min_y_initial:.3f}, {max_y_initial:.3f}]")
    
    print(f"  Grid: {args.grid_size}x{args.grid_size} = {args.grid_size**2} particles")
    print(f"  Device: {args.device}")
    print(f"  Ratchet friction: enabled")
    print()
    
    # =========================================================================
    # Create CPG
    # =========================================================================
    num_groups = (args.grid_size - 1) ** 2
    
    cpg = HopfCPG(
        num_groups=num_groups,
        frequency=args.frequency,
        amplitude=args.amplitude,
        direction=args.direction,
        coupling_strength=2.0,
        dt=args.dt,
    )
    
    print(f"CPG Configuration:")
    print(f"  Type: Hopf Oscillator (rate-coded)")
    print(f"  Groups: {num_groups}")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Direction: [{args.direction[0]:.2f}, {args.direction[1]:.2f}]")
    print()
    
    # =========================================================================
    # Create Force Injector
    # =========================================================================
    injector = BalloonForces(
        env.model,
        group_size=2,
        device=args.device,
        force_scale=args.force_scale,
    )
    
    print(f"Force Configuration:")
    print(f"  Scale: {args.force_scale}")
    print()
    
    # =========================================================================
    # Initialize Pygame and Renderer
    # =========================================================================
    pygame.init()
    
    # Create shared renderer for CPG overlay visualization
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Track body position for velocity calculation
    initial_positions = env.model.particle_q.numpy()
    initial_centroid_x = np.mean(initial_positions[:, 0])
    
    # Check initial Y position to set proper ground threshold
    min_y_initial = np.min(initial_positions[:, 1])
    max_y_initial = np.max(initial_positions[:, 1])
    print(f"\nBody initial position:")
    print(f"  Y range: [{min_y_initial:.3f}, {max_y_initial:.3f}]")
    print(f"  X centroid: {initial_centroid_x:.3f}")
    
    # Ground contact threshold: body is "on ground" if within 0.1 of its initial lowest point
    # This accounts for the body being centered in the box, not at y=0
    GROUND_CONTACT_THRESHOLD = min_y_initial + 0.1  # Just above initial position
    print(f"  Ground threshold: y < {GROUND_CONTACT_THRESHOLD:.3f}")
    
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
        
        # Get current positions from state (not model which has initial positions)
        current_positions = env.state_in.particle_q.numpy()
        injector.calculate_centroids(current_positions)
        
        # Check ground contact (for display only)
        min_y = np.min(current_positions[:, 1])
        body_on_ground = min_y < GROUND_CONTACT_THRESHOLD
        
        # Get CPG output
        cpg_output = cpg(t)
        
        # Clear forces
        injector.reset()
        
        # ALWAYS apply radial forces - they create internal deformation
        # Ground friction is what converts deformation into locomotion
        # NOTE: Don't multiply by force_scale here - injector handles it
        total_fx = 0.0
        for group_id in range(num_groups):
            cpg_val = cpg_output[group_id]
            if abs(cpg_val) > 0.001:
                injector.inject(group_id, cpg_val)
                total_fx += cpg_val
        
        # Get forces and step physics
        forces = injector.get_array()
        action = forces.flatten().astype(np.float32)
        
        # Debug: print CPG matrix every ~0.07s (quarter of 4Hz period) to see wave travel
        if frame_count % 7 == 1:
            max_force = np.max(np.abs(forces))
            nonzero = np.count_nonzero(forces)
            total_fx_actual = np.sum(forces[:, 0])
            total_fy_actual = np.sum(forces[:, 1])
            print(f"\n[CLASSIC CPG] t={t:.2f}s")
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
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate body velocity (using updated positions after step)
        # NOTE: Must use state_in, not model.particle_q (which is initial positions)
        new_positions = env.state_in.particle_q.numpy()
        current_centroid_x = np.mean(new_positions[:, 0])
        displacement = current_centroid_x - initial_centroid_x
        velocity = displacement / max(t, 0.01)
        
        # RECALCULATE centroids with CURRENT positions for drawing
        injector.calculate_centroids(new_positions)
        
        # RENDER: Draw scene + CPG overlays using shared Renderer
        env._sync_to_cpu()
        env._init_window()
        
        if env.window is not None:
            # Step 1: Draw base scene from env
            canvas = env._create_canvas()
            scale = env.window_height / env.boxsize
            env._draw_grid(canvas)
            env._draw_fem_triangles(canvas, scale)
            env._draw_springs(canvas, scale)
            env._draw_particles(canvas, scale)
            env._draw_centroids(canvas, scale)
            env._draw_ui_text(canvas)
            env._draw_legends(canvas)
            
            # Step 2: Draw CPG overlays using shared Renderer
            if injector.centroids is not None:
                group_side = int(np.sqrt(num_groups))
                
                # Draw group centroids with labels (hot pink)
                renderer.draw_group_centroids(canvas, injector.centroids)
                
                # Draw radial force arrows for each group (balloon inflate/deflate)
                for gid in range(num_groups):
                    if gid < len(injector.centroids):
                        renderer.draw_radial_force_arrows(
                            canvas,
                            injector.centroids[gid],
                            cpg_output[gid],
                            num_directions=4,
                        )
                
                # Draw CPG matrix display with direction indicator (bottom-right)
                renderer.draw_group_forces_matrix(
                    canvas,
                    cpg_output,
                    group_side,
                    title="CPG:",
                    direction=np.array(args.direction),
                )
            
            # Step 3: Blit and flip
            env.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
        
        # Progress output
        if frame_count % 100 == 0:
            fps = frame_count / max(time.time() - start_time, 0.01)
            print(f"t={t:.2f}s | disp={displacement:+.4f}m | vel={velocity:+.4f}m/s | "
                  f"ground={body_on_ground} | fx_sum={total_fx:.2f} | fps={fps:.1f}")
            # Debug: show centroids
            if injector.centroids is not None and env.window is not None:
                dbg_scale = env.window_height / args.boxsize
                print(f"  Groups: {num_groups}, Centroids shape: {injector.centroids.shape}")
                for gid in range(min(3, num_groups)):
                    cx, cy = injector.centroids[gid]
                    sx = int(cx * dbg_scale)
                    sy = int(env.window_height - cy * dbg_scale)
                    print(f"    Group {gid}: world({cx:.2f},{cy:.2f}) -> screen({sx},{sy})")
        
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
        print("  ✓ Body moved RIGHT - Locomotion working!")
    elif total_displacement < -0.01:
        print("  ← Body moved LEFT")
    else:
        print("  ✗ Body didn't move significantly")
        print("    Try: higher amplitude, more friction, higher force scale")
    
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

