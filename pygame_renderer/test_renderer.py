#!/usr/bin/env python3
"""
Test script for Renderer

Demonstrates all visualization features:
1. FEM / Spring strain legends
2. Tension visualization (springs + FEM colored by strain gradient)
3. Groups rendered in hot pink
4. External forces rendered by arrows
5. Optional SDF background with collision (brown=walls, white=passable)

Usage:
    python test_renderer.py                    # Physics demo
    python test_renderer.py --with-sdf         # With SDF background + collision
    python test_renderer.py --duration 30
    python test_renderer.py --no-fem
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'world_map'))

import numpy as np
import pygame
import argparse
import time

# Import the renderer
from renderer import Renderer

# Import physics components
try:
    import warp as wp
    from spring_mass_env import SpringMassEnv
    from sim import Model
    from solvers import SolverImplicit
    PHYSICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Physics modules not available ({e})")
    print("Running in visualization-only mode")
    PHYSICS_AVAILABLE = False

# Import SDF collision kernel
try:
    from world.kernels_sdf import apply_sdf_boundary_with_friction_2d
    SDF_COLLISION_AVAILABLE = True
except ImportError:
    SDF_COLLISION_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Test Renderer')
    parser.add_argument('--duration', '-t', type=float, default=30.0,
                        help='Duration in seconds (default: 30)')
    parser.add_argument('--grid-size', '-n', type=int, default=4,
                        help='Grid size NxN (default: 4)')
    parser.add_argument('--no-fem', action='store_true',
                        help='Disable FEM triangles')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--window-width', type=int, default=1000,
                        help='Window width (default: 1000)')
    parser.add_argument('--window-height', type=int, default=600,
                        help='Window height (default: 600)')
    parser.add_argument('--boxsize', type=float, default=2.5,
                        help='Simulation box size (default: 2.5)')
    # SDF background option
    parser.add_argument('--with-sdf', action='store_true',
                        help='Show SDF background (brown=walls, white=passable)')
    parser.add_argument('--map-image', type=str, default='map.png',
                        help='Path to world map image for SDF background (default: map.png)')
    return parser.parse_args()


def run_with_physics(args):
    """Run test with full physics simulation."""
    print("Running with physics simulation...")
    
    # Create environment
    env = SpringMassEnv(
        render_mode='human',
        N=args.grid_size,
        dt=0.01,
        spring_coeff=50.0,
        spring_damping=0.3,
        gravity=-0.5,
        boxsize=args.boxsize,
        device=args.device,
        with_fem=not args.no_fem,
        with_springs=True,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    
    # Use implicit solver for FEM
    if not args.no_fem:
        env.solver = SolverImplicit(
            env.model,
            dt=0.01,
            mass=1.0,
            preconditioner_type="diag",
            solver_type="bicgstab",
            max_iterations=30,
            tolerance=1e-3
        )
    
    env.reset(seed=42)
    
    # Create common renderer
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Load SDF background and collision if requested
    # SDF calculations use WorldMap from world_map module
    # Collision kernel uses apply_sdf_boundary_with_friction_2d from warp/world/
    sdf_surface = None
    sdf_collision_enabled = False
    world_map = None
    sdf_wp = None
    sdf_grad_x_wp = None
    sdf_grad_y_wp = None
    map_resolution = 1.0
    map_origin = np.array([0.0, 0.0])
    map_width = 0
    map_height = 0
    restitution = 0.3
    
    if args.with_sdf:
        # Use WorldMap from world_map module for all SDF calculations
        try:
            from world_map import WorldMap
            WORLD_MAP_AVAILABLE = True
        except ImportError:
            print("  Warning: WorldMap not available")
            WORLD_MAP_AVAILABLE = False
        
        if WORLD_MAP_AVAILABLE:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(args.map_image):
                image_path = args.map_image
            elif os.path.exists(os.path.join(script_dir, args.map_image)):
                image_path = os.path.join(script_dir, args.map_image)
            else:
                print(f"  Warning: Image '{args.map_image}' not found")
                image_path = None
            
            if image_path:
                print(f"  Loading SDF from: {image_path}")
                
                # Use WorldMap for SDF calculation (from world_map module)
                # Scale resolution to fit the simulation boxsize
                from PIL import Image
                img = Image.open(image_path)
                img_height = img.size[1]
                map_resolution = img_height / args.boxsize
                
                world_map = WorldMap(image_path=image_path, resolution=map_resolution)
                
                # Get SDF data from WorldMap
                map_data = world_map.to_numpy_arrays()
                sdf = map_data['sdf']
                sdf_grad_x = map_data['sdf_grad_x']
                sdf_grad_y = map_data['sdf_grad_y']
                map_origin = map_data['origin']
                map_width = sdf.shape[1]
                map_height = sdf.shape[0]
                
                print(f"  World size: {world_map.world_size[0]:.2f}x{world_map.world_size[1]:.2f}")
                
                # Create SDF surface for rendering (uses renderer)
                sdf_surface = renderer.create_sdf_surface(sdf, map_resolution, alpha=255, max_distance=20.0)
                
                # Setup collision arrays (Warp arrays for GPU kernel)
                if SDF_COLLISION_AVAILABLE:
                    sdf_wp = wp.array2d(sdf, dtype=float, device=args.device)
                    sdf_grad_x_wp = wp.array2d(sdf_grad_x, dtype=float, device=args.device)
                    sdf_grad_y_wp = wp.array2d(sdf_grad_y, dtype=float, device=args.device)
                    sdf_collision_enabled = True
                    print(f"  SDF collision: ENABLED")
                else:
                    print(f"  SDF collision: DISABLED (kernel not available)")
    
    # Setup group info for centroids
    N = args.grid_size
    groups_per_side = N - 1
    num_groups = groups_per_side * groups_per_side
    
    group_info = {}
    for group_id in range(num_groups):
        row = group_id // groups_per_side
        col = group_id % groups_per_side
        particles = [
            row * N + col,
            row * N + col + 1,
            (row + 1) * N + col,
            (row + 1) * N + col + 1,
        ]
        group_info[group_id] = particles
    
    # Pass group info to env for centroid calculation
    env.group_info = group_info
    
    print(f"Grid: {N}x{N} = {N*N} particles")
    print(f"Groups: {num_groups} ({groups_per_side}x{groups_per_side})")
    print(f"FEM: {'Enabled' if not args.no_fem else 'Disabled'}")
    print(f"SDF Background: {'Enabled' if sdf_surface else 'Disabled'}")
    print(f"SDF Collision: {'Enabled' if sdf_collision_enabled else 'Disabled'}")
    print()
    print("Controls:")
    print("  Q/ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset")
    print("  F     - Apply random force pulse")
    print()
    
    # Main loop
    pygame.init()
    t = 0.0
    dt = 0.01
    paused = False
    running = True
    frame_count = 0
    start_time = time.time()
    
    # Force state
    force_timer = 0.0
    force_duration = 0.5
    active_forces = {}
    
    while running and t < args.duration:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_r:
                    env.reset(seed=42)
                    t = 0.0
                    active_forces = {}
                    print("Reset!")
                elif event.key == pygame.K_f:
                    # Apply random force pulse to random groups
                    for gid in range(num_groups):
                        if np.random.random() > 0.5:
                            active_forces[gid] = np.random.uniform(-1.0, 1.0)
                    force_timer = force_duration
                    print(f"Force pulse applied to {len(active_forces)} groups")
        
        if paused:
            pygame.time.wait(50)
            continue
        
        # Update force timer
        if force_timer > 0:
            force_timer -= dt
            if force_timer <= 0:
                active_forces = {}
        
        # Create action from forces
        action = np.zeros(env.model.particle_count * 2, dtype=np.float32)
        force_scale = 20.0
        
        for gid, magnitude in active_forces.items():
            particles = group_info[gid]
            for pid in particles:
                # Radial force (simplified)
                action[pid * 2] += magnitude * force_scale * 0.25
                action[pid * 2 + 1] += magnitude * force_scale * 0.1
        
        # Step physics
        obs, reward, done, trunc, info = env.step(action)
        
        # Apply SDF collision if enabled
        if sdf_collision_enabled:
            forward_wp = wp.vec2(1.0, 0.0)  # Forward direction for ratchet friction
            wp.launch(
                kernel=apply_sdf_boundary_with_friction_2d,
                dim=env.model.particle_count,
                inputs=[
                    env.state_in.particle_q,
                    env.state_in.particle_qd,
                    sdf_wp,
                    sdf_grad_x_wp,
                    sdf_grad_y_wp,
                    map_resolution,
                    float(map_origin[0]),
                    float(map_origin[1]),
                    map_width,
                    map_height,
                    restitution,
                    forward_wp,
                    0,  # ratchet disabled
                ],
                device=env.model.device,
            )
        
        # Sync for rendering
        env._sync_to_cpu()
        
        # Calculate group centroids
        centroids = np.zeros((num_groups, 2), dtype=np.float32)
        for gid, particles in group_info.items():
            centroids[gid] = np.mean(env.pos_np[particles], axis=0)
        
        # ================================================================
        # CUSTOM RENDERING using Renderer
        # ================================================================
        env._init_window()
        
        if env.window is not None:
            # Create canvas
            canvas = renderer.create_canvas()
            
            # 0. Draw SDF background if enabled (FIRST - so everything draws on top)
            if sdf_surface is not None:
                renderer.draw_sdf_overlay(canvas, sdf_surface)
            else:
                # Draw background grid if no SDF
                renderer.draw_grid(canvas)
            
            # 1. Draw FEM triangles with strain coloring
            if not args.no_fem and env.tri_indices_np is not None:
                tri_strains = env.tri_strains_normalized_np
                renderer.draw_fem_triangles(
                    canvas,
                    env.tri_indices_np,
                    env.pos_np,
                    tri_strains
                )
            
            # 2. Draw springs with strain coloring
            spring_strains = env.spring_strains_normalized_np
            renderer.draw_springs(
                canvas,
                env.spring_indices_np,
                env.pos_np,
                spring_strains
            )
            
            # 3. Draw particles
            renderer.draw_particles(canvas, env.pos_np)
            
            # 4. Draw group centroids (HOT PINK)
            renderer.draw_group_centroids(
                canvas,
                centroids,
                group_ids=list(range(num_groups)),
                show_labels=True
            )
            
            # 5. Draw force arrows for active forces
            if active_forces:
                force_origins = []
                force_vectors = []
                for gid, mag in active_forces.items():
                    force_origins.append(centroids[gid])
                    # Radial force visualization (outward/inward)
                    force_vectors.append([mag * force_scale, mag * force_scale * 0.5])
                
                renderer.draw_force_arrows(
                    canvas,
                    np.array(force_origins),
                    np.array(force_vectors),
                    max_arrow_length=50.0,
                    force_scale=force_scale
                )
                
                # Also draw radial arrows on centroids with forces
                for gid, mag in active_forces.items():
                    renderer.draw_radial_force_arrows(
                        canvas,
                        centroids[gid],
                        mag,
                        num_directions=4,
                        arrow_scale=25.0
                    )
            
            # 6. Draw strain legends
            spring_scale = env.model.spring_strain_scale.numpy()[0] if hasattr(env.model, 'spring_strain_scale') else 0.01
            fem_scale = env.model.fem_strain_scale.numpy()[0] if hasattr(env.model, 'fem_strain_scale') else 0.01
            
            renderer.draw_strain_legends(
                canvas,
                spring_scale=spring_scale,
                fem_scale=fem_scale,
                show_fem=not args.no_fem
            )
            
            # 7. Draw force legend
            current_force = sum(abs(v) for v in active_forces.values()) * force_scale if active_forces else 0
            renderer.draw_force_legend(
                canvas,
                max_force=50.0,
                current_force=current_force
            )
            
            # 8. Draw info text
            fps = frame_count / max(time.time() - start_time, 0.01)
            info_lines = [
                (f"Time: {t:.2f}s | FPS: {fps:.1f}", (0, 0, 0)),
                (f"Grid: {N}x{N} | Groups: {num_groups} | Device: {args.device}", (100, 100, 100)),
                (f"Collision: {'ON' if sdf_collision_enabled else 'OFF'}", (0, 150, 0) if sdf_collision_enabled else (150, 150, 150)),
                (f"Active forces: {len(active_forces)} | Press F to apply", (0, 100, 200)),
            ]
            renderer.draw_info_text(canvas, info_lines, position=(10, 10))
            
            # 9. Draw group forces matrix (if forces active)
            if active_forces:
                force_values = np.array([active_forces.get(i, 0.0) for i in range(num_groups)])
                renderer.draw_group_forces_matrix(
                    canvas,
                    force_values,
                    groups_per_side,
                    center_group_id=num_groups // 2,
                    title="Forces"
                )
            
            # Blit and flip
            env.window.blit(canvas, canvas.get_rect())
            pygame.display.flip()
        
        t += dt
        frame_count += 1
        
        # Status output
        if frame_count % 100 == 0:
            fps = frame_count / max(time.time() - start_time, 0.01)
            print(f"t={t:.2f}s | fps={fps:.1f} | forces={len(active_forces)}")
    
    env.close()
    pygame.quit()
    print("\nTest complete!")


def run_visualization_only(args):
    """Run visualization demo without physics."""
    print("Running visualization-only demo...")
    
    pygame.init()
    
    # Create window
    window = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("Renderer Test (No Physics)")
    
    # Create renderer
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Load SDF background if requested (uses WorldMap from world_map module)
    sdf_surface = None
    if args.with_sdf:
        try:
            from world_map import WorldMap
            WORLD_MAP_AVAILABLE = True
        except ImportError:
            print("  Warning: WorldMap not available")
            WORLD_MAP_AVAILABLE = False
        
        if WORLD_MAP_AVAILABLE:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(args.map_image):
                image_path = args.map_image
            elif os.path.exists(os.path.join(script_dir, args.map_image)):
                image_path = os.path.join(script_dir, args.map_image)
            else:
                image_path = None
            
            if image_path:
                print(f"  Loading SDF background from: {image_path}")
                from PIL import Image
                img = Image.open(image_path)
                img_height = img.size[1]
                resolution = img_height / args.boxsize
                
                # Use WorldMap for SDF calculation
                world_map = WorldMap(image_path=image_path, resolution=resolution)
                sdf = world_map.sdf
                
                sdf_surface = renderer.create_sdf_surface(sdf, resolution, alpha=255, max_distance=20.0)
    
    # Create fake data
    N = args.grid_size
    groups_per_side = N - 1
    num_groups = groups_per_side * groups_per_side
    
    # Create grid positions
    spacing = 0.25
    offset_x = 1.0
    offset_y = 1.0
    
    positions = np.zeros((N * N, 2), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            positions[idx] = [offset_x + j * spacing, offset_y + i * spacing]
    
    # Create spring indices (horizontal + vertical + diagonal)
    spring_indices = []
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            # Right
            if j < N - 1:
                spring_indices.extend([idx, idx + 1])
            # Down
            if i < N - 1:
                spring_indices.extend([idx, idx + N])
            # Diagonal
            if i < N - 1 and j < N - 1:
                spring_indices.extend([idx, idx + N + 1])
                spring_indices.extend([idx + 1, idx + N])
    spring_indices = np.array(spring_indices, dtype=np.int32)
    num_springs = len(spring_indices) // 2
    
    # Create triangle indices
    tri_indices = []
    for i in range(N - 1):
        for j in range(N - 1):
            idx = i * N + j
            tri_indices.extend([idx, idx + 1, idx + N])
            tri_indices.extend([idx + 1, idx + N + 1, idx + N])
    tri_indices = np.array(tri_indices, dtype=np.int32)
    num_triangles = len(tri_indices) // 3
    
    # Create group centroids
    centroids = np.zeros((num_groups, 2), dtype=np.float32)
    group_info = {}
    for gid in range(num_groups):
        row = gid // groups_per_side
        col = gid % groups_per_side
        particles = [
            row * N + col,
            row * N + col + 1,
            (row + 1) * N + col,
            (row + 1) * N + col + 1,
        ]
        group_info[gid] = particles
        centroids[gid] = np.mean(positions[particles], axis=0)
    
    print(f"Grid: {N}x{N} = {N*N} particles")
    print(f"Springs: {num_springs}")
    print(f"Triangles: {num_triangles}")
    print(f"Groups: {num_groups}")
    print(f"SDF Background: {'Enabled' if sdf_surface else 'Disabled'}")
    print()
    print("Controls:")
    print("  Q/ESC - Quit")
    print("  F     - Toggle animated forces")
    print()
    
    # Main loop
    clock = pygame.time.Clock()
    t = 0.0
    running = True
    animate_forces = True
    
    while running and t < args.duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_f:
                    animate_forces = not animate_forces
                    print(f"Force animation: {'ON' if animate_forces else 'OFF'}")
        
        # Animate positions slightly
        animated_positions = positions.copy()
        for i in range(len(positions)):
            animated_positions[i, 0] += 0.02 * np.sin(t * 2 + i * 0.3)
            animated_positions[i, 1] += 0.02 * np.cos(t * 2 + i * 0.5)
        
        # Update centroids
        for gid, particles in group_info.items():
            centroids[gid] = np.mean(animated_positions[particles], axis=0)
        
        # Animated strains
        spring_strains = np.sin(t * 3 + np.arange(num_springs) * 0.2) * 0.8
        tri_strains = np.sin(t * 2 + np.arange(num_triangles) * 0.3) * 0.6
        
        # Animated forces
        if animate_forces:
            force_values = np.sin(t * 4 + np.arange(num_groups) * 0.5) * 0.8
            active_forces = {i: force_values[i] for i in range(num_groups) if abs(force_values[i]) > 0.3}
        else:
            active_forces = {}
            force_values = np.zeros(num_groups)
        
        # Create canvas
        canvas = renderer.create_canvas()
        
        # Draw SDF background if enabled
        if sdf_surface is not None:
            renderer.draw_sdf_overlay(canvas, sdf_surface)
        else:
            renderer.draw_grid(canvas)
        
        # Draw everything
        if not args.no_fem:
            renderer.draw_fem_triangles(canvas, tri_indices, animated_positions, tri_strains)
        
        renderer.draw_springs(canvas, spring_indices, animated_positions, spring_strains)
        renderer.draw_particles(canvas, animated_positions)
        renderer.draw_group_centroids(canvas, centroids, list(range(num_groups)))
        
        # Draw radial force arrows
        for gid, mag in active_forces.items():
            renderer.draw_radial_force_arrows(canvas, centroids[gid], mag, arrow_scale=20.0)
        
        # Draw legends
        renderer.draw_strain_legends(canvas, spring_scale=0.05, fem_scale=0.03, show_fem=not args.no_fem)
        renderer.draw_force_legend(canvas, max_force=1.0)
        
        # Draw group forces matrix
        renderer.draw_group_forces_matrix(canvas, force_values, groups_per_side, title="Forces")
        
        # Info text
        info_lines = [
            (f"Time: {t:.2f}s | Visualization Demo", (0, 0, 0)),
            (f"Grid: {N}x{N} | Springs: {num_springs} | Triangles: {num_triangles}", (100, 100, 100)),
            (f"Forces: {'Animated' if animate_forces else 'Off'} (Press F to toggle)", (0, 150, 0)),
        ]
        renderer.draw_info_text(canvas, info_lines, position=(10, 10))
        
        # Blit and flip
        window.blit(canvas, canvas.get_rect())
        pygame.display.flip()
        
        clock.tick(60)
        t += 1.0 / 60.0
    
    pygame.quit()
    print("\nTest complete!")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Renderer Test")
    print("=" * 60)
    print()
    
    if PHYSICS_AVAILABLE:
        run_with_physics(args)
    else:
        run_visualization_only(args)


if __name__ == "__main__":
    main()
