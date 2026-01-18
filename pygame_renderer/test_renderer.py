#!/usr/bin/env python3
"""
Test script for Renderer - Circle Soft Body

Demonstrates visualization of a circular soft body:
1. FEM / Spring strain legends
2. Tension visualization (springs + FEM colored by strain gradient)
3. Optional SDF background with collision (brown=walls, white=passable)

Usage:
    python test_renderer.py                    # Circle physics demo
    python test_renderer.py --with-sdf         # With SDF background + collision
    python test_renderer.py --radius 0.7       # Larger circle
    python test_renderer.py --num-boundary 30  # More boundary points
    python test_renderer.py --no-fem           # Springs only
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
    from models import CircleModel
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
    parser = argparse.ArgumentParser(description='Test Renderer - Circle Soft Body')
    parser.add_argument('--duration', '-t', type=float, default=30.0,
                        help='Duration in seconds (default: 30)')
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
    # Circle model options
    parser.add_argument('--radius', type=float, default=0.5,
                        help='Circle radius (default: 0.5)')
    parser.add_argument('--num-boundary', type=int, default=20,
                        help='Number of boundary points for circle (default: 20)')
    parser.add_argument('--num-rings', type=int, default=3,
                        help='Number of interior rings for circle (default: 3)')
    # SDF background option
    parser.add_argument('--with-sdf', action='store_true',
                        help='Show SDF background (brown=walls, white=passable)')
    parser.add_argument('--map-image', type=str, default='map.png',
                        help='Path to world map image for SDF background (default: map.png)')
    return parser.parse_args()


def run_with_physics(args):
    """Run test with full physics simulation."""
    print("Running with physics simulation...")
    
    # Create circle model
    print(f"\n🔵 Creating circular mesh model")
    print(f"   Radius: {args.radius}")
    print(f"   Boundary points: {args.num_boundary}")
    print(f"   Interior rings: {args.num_rings}")
    model = CircleModel(
        radius=args.radius,
        num_boundary=args.num_boundary,
        num_rings=args.num_rings,
        device=args.device,
        boxsize=args.boxsize,
        center=None  # Auto-center
    )
    print(f"✓ Created circle mesh with {model.particle_count} particles\n")
    
    # Create environment
    env = SpringMassEnv(
        render_mode='human',
        rows=1, cols=1,  # Not used for circle
        dt=0.01,
        spring_coeff=50.0,
        spring_damping=0.3,
        gravity=-0.5,
        boxsize=args.boxsize,
        device=args.device,
        model=model,
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
    
    # Numpy arrays for debug rendering
    sdf_np = None
    sdf_grad_x_np = None
    sdf_grad_y_np = None
    
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
                
                # Store numpy arrays for debug rendering
                sdf_np = sdf.copy()
                sdf_grad_x_np = sdf_grad_x.copy()
                sdf_grad_y_np = sdf_grad_y.copy()
                
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
    
    print(f"Circle: {env.model.particle_count} particles")
    print(f"FEM: {'Enabled' if not args.no_fem else 'Disabled'}")
    print(f"SDF Background: {'Enabled' if sdf_surface else 'Disabled'}")
    print(f"SDF Collision: {'Enabled' if sdf_collision_enabled else 'Disabled'}")
    print()
    print("Controls:")
    print("  Q/ESC - Quit")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset")
    print()
    
    # Main loop
    pygame.init()
    t = 0.0
    dt = 0.01
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
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_r:
                    env.reset(seed=42)
                    t = 0.0
                    print("Reset!")
        
        if paused:
            pygame.time.wait(50)
            continue
        
        # Step physics (no external forces)
        action = np.zeros(env.model.particle_count * 2, dtype=np.float32)
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
        
        # ================================================================
        # RENDERING using Renderer
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
            
            # 3. Draw particles (pink when near SDF collision, light blue otherwise)
            if sdf_collision_enabled and sdf_np is not None:
                particle_colors = renderer.compute_particle_collision_colors(
                    env.pos_np,
                    sdf_np,
                    map_resolution,
                    origin=(float(map_origin[0]), float(map_origin[1])),
                    near_threshold=0.1
                )
                renderer.draw_particles(canvas, env.pos_np, per_particle_colors=particle_colors)
            else:
                renderer.draw_particles(canvas, env.pos_np)
            
            # 4. Draw strain legends
            spring_scale = env.model.spring_strain_scale.numpy()[0] if hasattr(env.model, 'spring_strain_scale') else 0.01
            fem_scale = env.model.fem_strain_scale.numpy()[0] if hasattr(env.model, 'fem_strain_scale') else 0.01
            
            renderer.draw_strain_legends(
                canvas,
                spring_scale=spring_scale,
                fem_scale=fem_scale,
                show_fem=not args.no_fem
            )
            
            # 5. Draw info text
            fps = frame_count / max(time.time() - start_time, 0.01)
            info_lines = [
                (f"Time: {t:.2f}s | FPS: {fps:.1f}", (0, 0, 0)),
                (f"Circle: {env.model.particle_count} particles | Device: {args.device}", (100, 100, 100)),
                (f"Collision: {'ON' if sdf_collision_enabled else 'OFF'}", (0, 150, 0) if sdf_collision_enabled else (150, 150, 150)),
            ]
            renderer.draw_info_text(canvas, info_lines, position=(10, 10))
            
            # Blit and flip
            env.window.blit(canvas, canvas.get_rect())
            pygame.display.flip()
        
        t += dt
        frame_count += 1
        
        # Status output
        if frame_count % 100 == 0:
            fps = frame_count / max(time.time() - start_time, 0.01)
            print(f"t={t:.2f}s | fps={fps:.1f}")
    
    env.close()
    pygame.quit()
    print("\nTest complete!")


def run_visualization_only(args):
    """Run visualization demo without physics (simple circle animation)."""
    print("Running visualization-only demo (no physics)...")
    print("Note: Install physics modules for real simulation")
    
    pygame.init()
    
    # Create window
    window = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("Renderer Test - Circle (No Physics)")
    
    # Create renderer
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Generate circle mesh data (similar to CircleModel but numpy only)
    from scipy.spatial import Delaunay
    
    radius = args.radius
    num_boundary = args.num_boundary
    num_rings = args.num_rings
    center = (args.boxsize / 2.0, args.boxsize / 2.0)
    
    # Generate points
    all_pts = []
    angles = np.linspace(0, 2*np.pi, num_boundary, endpoint=False)
    all_pts.append(np.c_[radius*np.cos(angles), radius*np.sin(angles)])
    for ring in range(1, num_rings+1):
        r = radius * (num_rings-ring+1) / (num_rings+1)
        n = max(8, int(num_boundary * r / radius))
        a = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/num_boundary*ring
        all_pts.append(np.c_[r*np.cos(a), r*np.sin(a)])
    all_pts.append([[0.0, 0.0]])
    pts_norm = np.vstack(all_pts)
    
    # Delaunay triangulation
    tri = Delaunay(pts_norm)
    valid_tris = []
    for simplex in tri.simplices:
        cent = np.mean(pts_norm[simplex], axis=0)
        if np.linalg.norm(cent) <= radius * 1.01:
            p0, p1, p2 = pts_norm[simplex]
            cross = (p1-p0)[0]*(p2-p0)[1] - (p1-p0)[1]*(p2-p0)[0]
            if abs(cross) >= 1e-8:
                valid_tris.append(simplex)
    
    # Scale and center
    pts = pts_norm * radius * args.boxsize / 2.0
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    positions = pts.astype(np.float32)
    
    # Build springs from triangle edges
    edges = set()
    for t in valid_tris:
        for e in [(t[0],t[1]), (t[1],t[2]), (t[2],t[0])]:
            edges.add(tuple(sorted(e)))
    spring_indices = []
    for v0, v1 in edges:
        spring_indices.extend([v0, v1])
    spring_indices = np.array(spring_indices, dtype=np.int32)
    num_springs = len(edges)
    
    # Build triangle indices
    tri_indices = []
    for t in valid_tris:
        tri_indices.extend(t)
    tri_indices = np.array(tri_indices, dtype=np.int32)
    num_triangles = len(valid_tris)
    
    print(f"Circle: {len(positions)} particles")
    print(f"Springs: {num_springs}")
    print(f"Triangles: {num_triangles}")
    print()
    print("Controls: Q/ESC - Quit")
    print()
    
    # Main loop
    clock = pygame.time.Clock()
    t = 0.0
    running = True
    
    while running and t < args.duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
        
        # Animate positions slightly (fake jiggle)
        animated_positions = positions.copy()
        for i in range(len(positions)):
            animated_positions[i, 0] += 0.01 * np.sin(t * 3 + i * 0.5)
            animated_positions[i, 1] += 0.01 * np.cos(t * 3 + i * 0.7)
        
        # Animated strains
        spring_strains = np.sin(t * 3 + np.arange(num_springs) * 0.2) * 0.5
        tri_strains = np.sin(t * 2 + np.arange(num_triangles) * 0.3) * 0.4
        
        # Create canvas
        canvas = renderer.create_canvas()
        renderer.draw_grid(canvas)
        
        # Draw circle mesh
        if not args.no_fem:
            renderer.draw_fem_triangles(canvas, tri_indices, animated_positions, tri_strains)
        renderer.draw_springs(canvas, spring_indices, animated_positions, spring_strains)
        renderer.draw_particles(canvas, animated_positions)
        
        # Draw legends
        renderer.draw_strain_legends(canvas, spring_scale=0.05, fem_scale=0.03, show_fem=not args.no_fem)
        
        # Info text
        info_lines = [
            (f"Time: {t:.2f}s | Visualization Only (no physics)", (0, 0, 0)),
            (f"Circle: {len(positions)} particles | {num_springs} springs | {num_triangles} triangles", (100, 100, 100)),
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
