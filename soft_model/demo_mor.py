#!/usr/bin/env python3
"""
Model Order Reduction Demo

Complete MOR workflow:
1. TRAINING: Run full simulation, collect snapshots, compute POD basis
2. RUNNING: Simulate with reduced-order solver

Same visual setup as pygame_renderer/test_renderer.py:
- Circle soft body with FEM + springs
- SDF background (brown=walls, white=passable)
- SDF collision

Usage:
    python demo_mor.py                    # Full workflow (train + run)
    python demo_mor.py --train-only       # Training only
    python demo_mor.py --run-only         # Run with pre-trained model
    python demo_mor.py --compare          # Compare full vs reduced
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai-gym'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'world_map'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pygame_renderer'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_order_reduction'))

import numpy as np
import pygame
import argparse
import time

# Physics
import warp as wp
from sim import Model
from solvers import SolverImplicit

# MOR components
from model_order_reduction import PODReducer, SnapshotCollector
from reduction import SolverReduced, load_reduced_model

# Rendering
from renderer import Renderer

# SDF collision
try:
    from world.kernels_sdf import apply_sdf_boundary_with_friction_2d
    SDF_COLLISION_AVAILABLE = True
except ImportError:
    SDF_COLLISION_AVAILABLE = False
    print("Warning: SDF collision kernel not available")


def parse_args():
    parser = argparse.ArgumentParser(description='Model Order Reduction Demo')
    
    # Mode
    parser.add_argument('--train-only', action='store_true', help='Only run training')
    parser.add_argument('--run-only', action='store_true', help='Only run with reduced model')
    parser.add_argument('--compare', action='store_true', help='Compare full vs reduced')
    
    # Duration
    parser.add_argument('--train-duration', type=float, default=60.0,
                        help='Training duration (default: 60s)')
    parser.add_argument('--run-duration', type=float, default=30.0,
                        help='Running duration (default: 30s)')
    
    # Model
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--num-boundary', type=int, default=20)
    parser.add_argument('--num-rings', type=int, default=3)
    parser.add_argument('--boxsize', type=float, default=2.5)
    parser.add_argument('--no-fem', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    # MOR parameters
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='POD tolerance (default: 1e-3 = 99.9%% energy)')
    parser.add_argument('--snapshot-interval', type=int, default=5,
                        help='Collect snapshot every N steps')
    parser.add_argument('--output-dir', type=str, default='reduced_data/',
                        help='Output directory for reduced model')
    
    # Rendering
    parser.add_argument('--window-width', type=int, default=1000)
    parser.add_argument('--window-height', type=int, default=600)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--with-sdf', action='store_true', default=True)
    parser.add_argument('--no-sdf', action='store_true')
    parser.add_argument('--map-image', type=str, default='map.png')
    
    return parser.parse_args()


def load_sdf_data(args, renderer):
    """Load SDF for rendering and collision."""
    if args.no_sdf:
        return None, None, None
    
    try:
        from world_map import WorldMap
    except ImportError:
        print("  WorldMap not available")
        return None, None, None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        args.map_image,
        os.path.join(script_dir, args.map_image),
        os.path.join(script_dir, '..', 'pygame_renderer', 'map.png'),
        '/workspace/pygame_renderer/map.png',
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        print("  map.png not found, SDF disabled")
        return None, None, None
    
    print(f"  Loading SDF from: {image_path}")
    
    from PIL import Image
    img = Image.open(image_path)
    img_height = img.size[1]
    map_resolution = img_height / args.boxsize
    
    world_map = WorldMap(image_path=image_path, resolution=map_resolution)
    map_data = world_map.to_numpy_arrays()
    
    # Create SDF surface for rendering
    sdf_surface = renderer.create_sdf_surface(
        map_data['sdf'], map_resolution, alpha=255, max_distance=20.0
    )
    
    # Create Warp arrays for collision
    sdf_collision = None
    if SDF_COLLISION_AVAILABLE:
        sdf_collision = {
            'sdf': wp.array2d(map_data['sdf'], dtype=float, device=args.device),
            'sdf_grad_x': wp.array2d(map_data['sdf_grad_x'], dtype=float, device=args.device),
            'sdf_grad_y': wp.array2d(map_data['sdf_grad_y'], dtype=float, device=args.device),
            'resolution': map_resolution,
            'origin': map_data['origin'],
            'width': map_data['sdf'].shape[1],
            'height': map_data['sdf'].shape[0],
            'sdf_np': map_data['sdf'].copy(),
        }
        print(f"  SDF collision: ENABLED")
    
    return sdf_surface, sdf_collision, map_resolution


def apply_sdf_collision(model, state, sdf_collision):
    """Apply SDF collision to state."""
    if sdf_collision is None or not SDF_COLLISION_AVAILABLE:
        return
    
    wp.launch(
        kernel=apply_sdf_boundary_with_friction_2d,
        dim=model.particle_count,
        inputs=[
            state.particle_q,
            state.particle_qd,
            sdf_collision['sdf'],
            sdf_collision['sdf_grad_x'],
            sdf_collision['sdf_grad_y'],
            sdf_collision['resolution'],
            float(sdf_collision['origin'][0]),
            float(sdf_collision['origin'][1]),
            sdf_collision['width'],
            sdf_collision['height'],
            0.3,  # restitution
            wp.vec2(1.0, 0.0),  # forward direction
            0,  # ratchet disabled
        ],
        device=model.device,
    )


def train(args):
    """Training phase: collect snapshots and compute POD basis."""
    print("\n" + "=" * 60)
    print("  TRAINING PHASE - Collecting Snapshots")
    print("=" * 60)
    
    wp.init()
    
    # Create model
    model = Model.from_circle(
        radius=args.radius,
        num_boundary=args.num_boundary,
        num_rings=args.num_rings,
        device=args.device,
        boxsize=args.boxsize,
    )
    
    # Use SAME physics as test_renderer.py (Model.from_circle defaults)
    # test_renderer.py passes model to SpringMassEnv, which IGNORES spring_coeff/damping/gravity
    # So the actual physics are the defaults from Model.from_circle():
    # - spring_stiffness = 40.0
    # - spring_damping = 0.5
    # - gravity = -0.1
    # DO NOT OVERRIDE THESE - they're already set correctly by from_circle()
    
    print(f"\n  Model: {model.particle_count} particles, {model.spring_count} springs")
    print(f"  Physics: stiffness=40 (default), damping=0.5 (default), gravity=-0.1 (default)")
    print(f"  Full DOF: {model.particle_count * 2}")
    
    # Create solver (same as test_renderer.py)
    solver = SolverImplicit(
        model, dt=0.01, mass=1.0,
        preconditioner_type="diag",
        solver_type="bicgstab",
        max_iterations=30,
        tolerance=1e-3
    )
    
    # Create renderer for visualization
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Load SDF
    sdf_surface, sdf_collision, _ = load_sdf_data(args, renderer)
    
    # Initialize snapshot collector
    collector = SnapshotCollector(snapshot_interval=args.snapshot_interval)
    
    # Initialize states
    state_in = model.state()
    state_out = model.state()
    
    # Set rest position
    rest_pos = state_in.particle_q.numpy()
    collector.set_rest_position(rest_pos)
    
    # Setup pygame if rendering
    screen = None
    clock = None
    if not args.no_render:
        pygame.init()
        screen = pygame.display.set_mode((args.window_width, args.window_height))
        pygame.display.set_caption("MOR Training - Collecting Snapshots")
        clock = pygame.time.Clock()
    
    # Training loop
    dt = 0.01
    n_steps = int(args.train_duration / dt)
    t = 0.0
    running = True
    
    print(f"\n  Running {n_steps} steps ({args.train_duration}s)...")
    print(f"  Snapshot interval: every {args.snapshot_interval} steps")
    start_time = time.time()
    
    for step in range(n_steps):
        if not args.no_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break
        
        # Physics step (gravity only - same as pygame_renderer)
        solver.step(state_in, state_out, dt)
        
        # Apply SDF collision
        apply_sdf_collision(model, state_out, sdf_collision)
        
        # Collect snapshot with velocities (Cotangent Lift for better force retention)
        if step % args.snapshot_interval == 0:
            pos = state_out.particle_q.numpy()
            vel = state_out.particle_qd.numpy()
            collector.add_snapshot(pos, velocities=vel)
        
        # Render (same style as running mode)
        if not args.no_render and step % 2 == 0:
            canvas = renderer.create_canvas()
            
            if sdf_surface is not None:
                renderer.draw_sdf_overlay(canvas, sdf_surface)
            else:
                renderer.draw_grid(canvas)
            
            pos_np = state_out.particle_q.numpy()
            
            # Draw FEM with strain coloring
            if model.tri_count > 0:
                tri_strains = model.tri_strains_normalized.numpy() if hasattr(model, 'tri_strains_normalized') else None
                renderer.draw_fem_triangles(canvas, model.tri_indices.numpy(), pos_np, tri_strains)
            
            # Draw springs with strain coloring
            spring_strains = model.spring_strains_normalized.numpy() if hasattr(model, 'spring_strains_normalized') else None
            renderer.draw_springs(canvas, model.spring_indices.numpy(), pos_np, spring_strains)
            
            # Draw particles with collision coloring
            if sdf_collision and sdf_collision.get('sdf_np') is not None:
                colors = renderer.compute_particle_collision_colors(
                    pos_np, sdf_collision['sdf_np'], sdf_collision['resolution'],
                    origin=(float(sdf_collision['origin'][0]), float(sdf_collision['origin'][1]))
                )
                renderer.draw_particles(canvas, pos_np, per_particle_colors=colors)
            else:
                renderer.draw_particles(canvas, pos_np)
            
            # Draw strain legends
            spring_scale = model.spring_strain_scale.numpy()[0] if hasattr(model, 'spring_strain_scale') else 0.01
            fem_scale = model.fem_strain_scale.numpy()[0] if hasattr(model, 'fem_strain_scale') else 0.01
            renderer.draw_strain_legends(canvas, spring_scale=spring_scale, fem_scale=fem_scale, show_fem=model.tri_count > 0)
            
            # Info text
            fps = clock.get_fps()
            renderer.draw_info_text(canvas, [
                (f"TRAINING: {step}/{n_steps} ({100*step/n_steps:.0f}%) | FPS: {fps:.0f}", (0, 0, 0)),
                (f"Snapshots: {collector.n_snapshots}", (100, 100, 100)),
                (f"Collision: {'ON' if sdf_collision else 'OFF'}", (0, 150, 0) if sdf_collision else (150, 150, 150)),
            ], position=(10, 10))
            
            screen.blit(canvas, (0, 0))
            pygame.display.flip()
            clock.tick(60)
        
        state_in, state_out = state_out, state_in
        t += dt
    
    elapsed = time.time() - start_time
    
    if not args.no_render:
        pygame.quit()
    
    print(f"\n  Collected {collector.n_snapshots} snapshots in {elapsed:.1f}s")
    
    # Compute POD basis
    print("\n" + "=" * 60)
    print("  COMPUTING POD BASIS")
    print("=" * 60)
    
    pod_reducer = PODReducer(
        tolerance=args.tolerance,
        verbose=True
    )
    
    snapshots = collector.get_snapshots()
    basis, pod_info = pod_reducer.fit(snapshots, collector.rest_position)
    
    print(f"\n  Reduced: {basis.n_full} DOF → {basis.n_modes} modes")
    print(f"  Compression: {basis.compression_ratio:.1f}x")
    print(f"  Energy captured: {pod_info['energy_captured']*100:.2f}%")
    
    # Save reduced model
    os.makedirs(args.output_dir, exist_ok=True)
    basis.save(args.output_dir, "reduced_basis")
    
    # Save metadata
    import json
    metadata = {
        'n_particles': model.particle_count,
        'n_modes': basis.n_modes,
        'tolerance': args.tolerance,
        'n_snapshots': collector.n_snapshots,
        'energy_captured': float(pod_info['energy_captured']),
        'model': {
            'radius': args.radius,
            'num_boundary': args.num_boundary,
            'num_rings': args.num_rings,
            'boxsize': args.boxsize,
        }
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {args.output_dir}")
    print("=" * 60)
    
    return basis


def run_reduced(args):
    """Running phase: simulate with reduced-order solver."""
    print("\n" + "=" * 60)
    print("  RUNNING PHASE - Reduced-Order Simulation")
    print("=" * 60)
    
    wp.init()
    
    # Load reduced model
    if not os.path.exists(args.output_dir):
        print(f"Error: Reduced model not found in {args.output_dir}")
        print("Run with --train-only first")
        return
    
    reduced_data = load_reduced_model(args.output_dir)
    
    print(f"\n  Loaded reduced model:")
    print(f"    Modes: {reduced_data['n_modes']}")
    print(f"    Particles: {reduced_data['n_particles']}")
    
    # Create model (must match training)
    model = Model.from_circle(
        radius=args.radius,
        num_boundary=args.num_boundary,
        num_rings=args.num_rings,
        device=args.device,
        boxsize=args.boxsize,
    )
    
    # Use SAME physics as test_renderer.py (Model.from_circle defaults)
    # DO NOT override - defaults are already set correctly by from_circle()
    
    # Create reduced solver
    solver = SolverReduced(
        model=model,
        reduced_data=reduced_data,
        mass=1.0,
    )
    
    stats = solver.get_compression_stats()
    print(f"    Compression: {stats['compression_ratio']:.1f}x")
    print(f"    Expected speedup: ~{stats['linear_solve_speedup']:.0f}x")
    
    # Create renderer
    renderer = Renderer(
        window_width=args.window_width,
        window_height=args.window_height,
        boxsize=args.boxsize,
    )
    
    # Load SDF
    sdf_surface, sdf_collision, _ = load_sdf_data(args, renderer)
    
    # Initialize state
    state_in = model.state()
    state_out = model.state()
    
    # Setup pygame
    if args.no_render:
        return
    
    pygame.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("MOR Running - Reduced Solver")
    clock = pygame.time.Clock()
    
    # Run simulation
    dt = 0.01
    t = 0.0
    running = True
    frame_count = 0
    start_time = time.time()
    
    print(f"\n  Running reduced simulation for {args.run_duration}s...")
    print("  Controls: Q/ESC=quit, SPACE=pause, R=reset")
    
    paused = False
    
    while running and t < args.run_duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state_in = model.state()
                    state_out = model.state()
                    t = 0.0
        
        if paused:
            pygame.time.wait(50)
            continue
        
        # Step reduced solver
        solver.step(state_in, state_out, dt)
        
        # Apply SDF collision
        apply_sdf_collision(model, state_out, sdf_collision)
        
        # Render
        canvas = renderer.create_canvas()
        
        if sdf_surface is not None:
            renderer.draw_sdf_overlay(canvas, sdf_surface)
        else:
            renderer.draw_grid(canvas)
        
        pos_np = state_out.particle_q.numpy()
        
        if model.tri_count > 0:
            tri_strains = model.tri_strains_normalized.numpy() if hasattr(model, 'tri_strains_normalized') else None
            renderer.draw_fem_triangles(canvas, model.tri_indices.numpy(), pos_np, tri_strains)
        
        spring_strains = model.spring_strains_normalized.numpy() if hasattr(model, 'spring_strains_normalized') else None
        renderer.draw_springs(canvas, model.spring_indices.numpy(), pos_np, spring_strains)
        
        if sdf_collision and sdf_collision.get('sdf_np') is not None:
            colors = renderer.compute_particle_collision_colors(
                pos_np, sdf_collision['sdf_np'], sdf_collision['resolution'],
                origin=(float(sdf_collision['origin'][0]), float(sdf_collision['origin'][1]))
            )
            renderer.draw_particles(canvas, pos_np, per_particle_colors=colors)
        else:
            renderer.draw_particles(canvas, pos_np)
        
        # Draw strain legends
        spring_scale = model.spring_strain_scale.numpy()[0] if hasattr(model, 'spring_strain_scale') else 0.01
        fem_scale = model.fem_strain_scale.numpy()[0] if hasattr(model, 'fem_strain_scale') else 0.01
        renderer.draw_strain_legends(canvas, spring_scale=spring_scale, fem_scale=fem_scale, show_fem=model.tri_count > 0)
        
        fps = frame_count / max(time.time() - start_time, 0.01)
        renderer.draw_info_text(canvas, [
            (f"REDUCED SOLVER | t={t:.2f}s | FPS: {fps:.0f}", (0, 100, 0)),
            (f"Modes: {stats['n_reduced']} / {stats['n_full']} DOF", (100, 100, 100)),
            (f"Compression: {stats['compression_ratio']:.1f}x", (100, 100, 100)),
            (f"Collision: {'ON' if sdf_collision else 'OFF'}", (0, 150, 0) if sdf_collision else (150, 150, 150)),
        ], position=(10, 10))
        
        screen.blit(canvas, (0, 0))
        pygame.display.flip()
        clock.tick(60)
        
        state_in, state_out = state_out, state_in
        t += dt
        frame_count += 1
    
    pygame.quit()
    
    elapsed = time.time() - start_time
    print(f"\n  Completed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.0f} FPS)")
    print("=" * 60)


def compare(args):
    """Compare full-order vs reduced-order solver."""
    print("\n" + "=" * 60)
    print("  COMPARISON: Full vs Reduced Solver")
    print("=" * 60)
    
    wp.init()
    
    # Load reduced model
    if not os.path.exists(args.output_dir):
        print(f"Reduced model not found. Training first...")
        train(args)
    
    reduced_data = load_reduced_model(args.output_dir)
    
    # Create model
    model = Model.from_circle(
        radius=args.radius,
        num_boundary=args.num_boundary,
        num_rings=args.num_rings,
        device=args.device,
        boxsize=args.boxsize,
    )
    
    # Use SAME physics as test_renderer.py (Model.from_circle defaults)
    # DO NOT override - defaults are already set correctly by from_circle()
    
    n_steps = 500
    warmup = 50
    dt = 0.01
    
    print(f"\n  Benchmarking {n_steps} steps (warmup: {warmup})...")
    print(f"  Physics: stiffness=40 (default), damping=0.5 (default), gravity=-0.1 (default)")
    
    # Full solver benchmark
    print("\n  --- Full Solver ---")
    solver_full = SolverImplicit(model, dt=dt, mass=1.0)
    state_in = model.state()
    state_out = model.state()
    
    for _ in range(warmup):
        solver_full.step(state_in, state_out, dt)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    
    full_times = []
    start = time.time()
    for _ in range(n_steps):
        t0 = time.time()
        solver_full.step(state_in, state_out, dt)
        wp.synchronize()
        full_times.append(time.time() - t0)
        state_in, state_out = state_out, state_in
    full_elapsed = time.time() - start
    full_avg = np.mean(full_times) * 1000
    
    print(f"  Time: {full_elapsed:.2f}s, Avg: {full_avg:.3f}ms/step")
    
    # Reduced solver benchmark
    print("\n  --- Reduced Solver ---")
    solver_reduced = SolverReduced(model, reduced_data, mass=1.0)
    state_in = model.state()
    state_out = model.state()
    
    for _ in range(warmup):
        solver_reduced.step(state_in, state_out, dt)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    
    reduced_times = []
    start = time.time()
    for _ in range(n_steps):
        t0 = time.time()
        solver_reduced.step(state_in, state_out, dt)
        wp.synchronize()
        reduced_times.append(time.time() - t0)
        state_in, state_out = state_out, state_in
    reduced_elapsed = time.time() - start
    reduced_avg = np.mean(reduced_times) * 1000
    
    print(f"  Time: {reduced_elapsed:.2f}s, Avg: {reduced_avg:.3f}ms/step")
    
    # Results
    speedup = full_avg / reduced_avg
    stats = solver_reduced.get_compression_stats()
    
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Model: {model.particle_count} particles, {model.spring_count} springs, {model.tri_count} triangles")
    print(f"  Full DOF: {stats['n_full']}")
    print(f"  Reduced DOF: {stats['n_reduced']}")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Full solver: {full_avg:.3f} ms/step")
    print(f"  Reduced solver: {reduced_avg:.3f} ms/step")
    if speedup >= 1.0:
        print(f"  Speedup: {speedup:.2f}x FASTER")
    else:
        print(f"  Speedup: {1/speedup:.2f}x SLOWER (model too small for MOR)")
    print("=" * 60)
    print("")
    if speedup < 1.0:
        print("  NOTE: MOR provides speedup for larger models (200+ particles).")
        print("        For small models, full GPU solver is faster.")
    else:
        print("  SUCCESS: MOR provides speedup for this model size!")
    print("")


def main():
    args = parse_args()
    
    if args.no_sdf:
        args.with_sdf = False
    
    print("=" * 60)
    print("  Model Order Reduction Demo")
    print("=" * 60)
    
    if args.compare:
        compare(args)
    elif args.run_only:
        run_reduced(args)
    elif args.train_only:
        train(args)
    else:
        # Full workflow
        train(args)
        run_reduced(args)


if __name__ == "__main__":
    main()
