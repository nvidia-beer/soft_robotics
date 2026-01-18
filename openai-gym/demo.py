#!/usr/bin/env python3
"""
Demo script for Spring Mass Environment
Using modular solver architecture (Model + Solver pattern)
Supports both semi-implicit (default) and implicit solvers
"""

import sys
import os
# Add warp directory to path (sibling directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))

import argparse
import numpy as np
from spring_mass_env import SpringMassEnv
from solvers import SolverImplicit, SolverImplicitFEM, SolverVBD
from models import TessellationModel, CircleModel


def main():
    """Run spring mass simulation with modular architecture"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Spring Mass System Demo")
    parser.add_argument('--implicit', action='store_true', 
                       help='Use implicit solver with FEM (unconditionally stable)')
    parser.add_argument('--implicit-fem', action='store_true',
                       help='Use FULLY implicit FEM solver (FEM stiffness in system matrix)')
    parser.add_argument('--vbd', action='store_true',
                       help='Use VBD solver (Vertex Block Descent - parallel GPU optimization)')
    parser.add_argument('--no-fem', action='store_true',
                       help='Disable FEM triangles (implicit solver only, spring-only simulation)')
    parser.add_argument('--no-springs', action='store_true',
                       help='Disable springs (FEM-only simulation, requires --implicit)')
    parser.add_argument('--rows', type=int, default=5, 
                       help='Grid rows (height, ignored if --tessellation is provided)')
    parser.add_argument('--cols', type=int, default=5, 
                       help='Grid cols (width, ignored if --tessellation is provided)')
    parser.add_argument('--tessellation', type=str, default=None,
                       help='Path to tessellation JSON file (e.g., /workspace/tessellation/model.json)')
    parser.add_argument('--circle', action='store_true',
                       help='Use circular mesh model (procedurally generated)')
    parser.add_argument('--radius', type=float, default=0.75,
                       help='Circle radius for --circle mode (default: 0.75)')
    parser.add_argument('--num-boundary', type=int, default=16,
                       help='Number of boundary points for --circle mode (default: 16)')
    parser.add_argument('--num-rings', type=int, default=3,
                       help='Number of interior rings for --circle mode (default: 3)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Time step (auto: 0.1 for semi-implicit, 0.2 for implicit)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of simulation steps')
    parser.add_argument('--no-render', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--window-width', type=int, default=1000,
                       help='Window width in pixels (default: 1000)')
    parser.add_argument('--window-height', type=int, default=500,
                       help='Window height in pixels (default: 500)')
    parser.add_argument('--boxsize', type=float, default=2.5,
                       help='Bounding box size (height, default: 2.5)')
    parser.add_argument('--spring-stiffness', type=float, default=40.0,
                       help='Spring stiffness coefficient (default: 40.0)')
    parser.add_argument('--spring-damping', type=float, default=0.5,
                       help='Spring damping coefficient (default: 0.5)')
    parser.add_argument('--fem-young', type=float, default=50.0,
                       help='FEM Young\'s modulus E (default: 50.0, 0=disable FEM)')
    parser.add_argument('--fem-poisson', type=float, default=0.3,
                       help='FEM Poisson ratio nu (default: 0.3)')
    parser.add_argument('--fem-damping', type=float, default=2.0,
                       help='FEM damping coefficient (default: 2.0)')
    parser.add_argument('--profile-render', action='store_true',
                       help='Enable render profiling (shows timing breakdown every 100 frames)')
    parser.add_argument('--profile-physics', action='store_true',
                       help='Enable physics profiling (shows GPU/CPU timing every 100 steps)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Computation device')
    args = parser.parse_args()
    
    # Set default timestep based on solver
    # Fully implicit FEM can handle larger timesteps
    if args.dt is None:
        if args.vbd:
            args.dt = 0.01  # VBD uses smaller timesteps for iterative convergence
        elif args.implicit_fem:
            args.dt = 0.05  # Fully implicit FEM handles larger timesteps
        elif args.implicit:
            args.dt = 0.05  # Implicit solver
        else:
            args.dt = 0.05  # Semi-implicit
    
    # Create environment (uses Model + Solver internally)
    render_mode = None if args.no_render else 'human'
    
    # FEM only makes sense with implicit solver or VBD (semi-implicit doesn't use it)
    use_fem = (args.implicit or args.implicit_fem or args.vbd) and not args.no_fem
    
    # Use spring parameters from command line
    spring_coeff = args.spring_stiffness
    spring_damping = args.spring_damping
    
    # If FEM is enabled, reduce spring stiffness to avoid double-stiffness
    if use_fem:
        spring_coeff = spring_coeff * 0.25  # 25% of original when FEM is on
        print(f"  Note: Reducing spring stiffness to {spring_coeff:.1f} (25%) to complement FEM")
    
    # Load model from tessellation or create procedural model
    model = None
    if args.tessellation:
        # Resolve path relative to workspace root if needed
        tess_path = args.tessellation
        if not os.path.isabs(tess_path):
            # If relative, resolve from workspace root (parent of openai-gym)
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tess_path = os.path.join(workspace_root, tess_path)
        
        print(f"\n📐 Loading tessellation from: {tess_path}")
        model = TessellationModel(
            json_path=tess_path,
            device=args.device,
            boxsize=args.boxsize,
            scale=0.8
        )
        print(f"✓ Loaded tessellation with {model.particle_count} particles\n")
    elif args.circle:
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
    
    env = SpringMassEnv(
        render_mode=render_mode,
        rows=args.rows, cols=args.cols,
        dt=args.dt,
        spring_coeff=spring_coeff,
        spring_damping=spring_damping,
        gravity=-0.1,
        boxsize=args.boxsize,
        device=args.device,
        model=model,  # Pass tessellation model if loaded
        with_fem=use_fem,  # Enable FEM only for implicit solver
        with_springs=not args.no_springs,  # Enable springs unless --no-springs is specified
        window_width=args.window_width,  # Adjustable window width
        window_height=args.window_height  # Adjustable window height
    )
    
    # Enable profiling if requested
    if args.profile_render:
        env._profile_render = True
        print("\n🔍 Render profiling ENABLED (will show timing every 100 frames)\n")
    
    if args.profile_physics:
        env._profile_physics = True
        print("\n🔍 Physics profiling ENABLED (will show GPU/CPU timing every 100 steps)\n")
    
    # Replace solver with implicit or VBD if requested
    if args.implicit or args.implicit_fem or args.vbd:
        # Update FEM parameters if FEM is enabled
        if use_fem and hasattr(env.model, 'tri_materials') and env.model.tri_materials is not None:
            # Recalculate Lame parameters from user-provided values
            E = args.fem_young
            nu = args.fem_poisson
            k_damp = args.fem_damping
            
            k_mu = E / (2.0 * (1.0 + nu))
            k_lambda = E * nu / (1.0 - nu * nu)
            
            # Update all triangles with new material properties
            import warp as wp
            env.model.tri_materials.fill_(wp.vec3(k_mu, k_lambda, k_damp))
            print(f"  FEM params: E={E}, nu={nu}, damp={k_damp}")
        
        if args.vbd:
            print("\n🔷 Switching to VBD solver (Vertex Block Descent)...")
            print("   (GPU-parallel optimization-based implicit integration)")
            # Match implicit FEM: use particle_mass=1.0 equivalent
            # VBD computes mass = density * area, so we need to scale density
            # to get similar effective mass per particle
            avg_area = env.model.tri_areas.numpy().mean() if hasattr(env.model, 'tri_areas') else 0.02
            avg_incident = 6  # typical interior vertex has ~6 incident triangles
            # effective_mass_per_particle ≈ avg_incident * (density * avg_area / 3)
            # To match implicit's mass=1.0: density = 3 / (avg_incident * avg_area)
            target_mass = 1.0  # Same as implicit solver default
            vbd_density = (3.0 * target_mass) / (avg_incident * avg_area)
            
            env.solver = SolverVBD(
                env.model,
                dt=args.dt,
                dx_tol=1e-5,
                max_iter=20,
                damping_coefficient=1.0,  # Use material k_damp directly
                density=vbd_density
            )
            print(f"   Density: {vbd_density:.2f} (target mass ~{target_mass}), max_iter: 20")
        elif args.implicit_fem:
            print("\n⚡⚡ Switching to FULLY IMPLICIT FEM solver...")
            print("   (FEM tangent stiffness included in system matrix)")
            env.solver = SolverImplicitFEM(
                env.model,
                dt=args.dt,
                mass=1.0,
                preconditioner_type="diag",
                solver_type="bicgstab",
                max_iterations=30,
                tolerance=1e-3,
                rebuild_matrix_every=1  # Rebuild every step for nonlinear accuracy
            )
        else:
            print("\n⚡ Switching to IMPLICIT solver...")
            env.solver = SolverImplicit(
                env.model,
                dt=args.dt,
                mass=1.0,
                preconditioner_type="diag",
                solver_type="bicgstab",
                max_iterations=30,  # More iterations for FEM stability
                tolerance=1e-3      # Slightly looser tolerance for speed
            )
    
    if args.vbd:
        solver_name = "VBD (Vertex Block Descent - GPU Parallel)"
    elif args.implicit_fem:
        solver_name = "Fully Implicit FEM (K_fem in system matrix)"
    elif args.implicit:
        solver_name = "Implicit (Unconditionally Stable)"
    else:
        solver_name = "Semi-Implicit"
    
    print("=" * 60)
    print(f"Spring Mass System Demo - {solver_name}")
    print("=" * 60)
    print(f"Grid size: {env.cols}x{env.rows} = {env.model.particle_count} nodes")
    print(f"Number of springs: {env.model.spring_count}")
    print(f"Spring params: k={spring_coeff}, d={spring_damping}")
    
    # Check if FEM is present
    has_fem = hasattr(env.model, 'tri_count') and env.model.tri_count > 0
    if has_fem:
        print(f"FEM triangles: {env.model.tri_count}")
    else:
        print(f"FEM triangles: 0 (spring-only simulation)")
    
    print(f"Timestep: {args.dt}s", end="")
    if args.implicit:
        print(" ⚡ (stable with implicit solver)")
    else:
        print()
    print(f"Device: {args.device}")
    print(f"Visualization: {'Yes' if not args.no_render else 'No'}")
    if not args.no_render:
        print(f"  - Window size: {args.window_width}×{args.window_height}")
        print(f"  - FPS: Unlimited (maximum speed)")
    print("=" * 60)
    print("\nArchitecture:")
    print("  - sim.Model: Static system description")
    print("  - sim.State: Time-varying state")
    if args.vbd:
        print("  - solvers.SolverVBD: Vertex Block Descent 🔷")
        print("  - GPU-parallel optimization (graph coloring)")
        print("  - Per-vertex 2x2 Newton solve (no global assembly)")
        print("  - Near-linear scaling with mesh size")
        print("  - Stable Neo-Hookean material model")
        print(f"  - Using dt={args.dt}s (iterative convergence)")
    elif args.implicit_fem:
        print("  - solvers.SolverImplicitFEM: Fully implicit FEM ⚡⚡")
        print("  - FEM tangent stiffness in system matrix: A = M - hD - h²(K_spring + K_fem)")
        print("  - Better stability for stiff FEM materials")
        print("  - Matrix rebuilt every step for nonlinear accuracy")
        print(f"  - Using dt={args.dt}s")
    elif args.implicit:
        print("  - solvers.SolverImplicit: Implicit time integration ⚡")
        print("  - Unconditionally stable (better for stiff systems)")
        print("  - Sparse BSR matrices (2x2 blocks)")
        print("  - BiCGStab iterative solver (20 iterations)")
        print(f"  - Using dt={args.dt}s")
    else:
        print("  - solvers.SolverSemiImplicit: Semi-implicit time integration")
        print(f"  - Using dt={args.dt}s")
    print("  - Follows Newton's modular pattern")
    print("=" * 60)
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial energy: {info['total_energy']:.4f}")
    
    if args.vbd:
        print("\n💡 VBD (Vertex Block Descent) solver features:")
        print(f"   ✓ Using dt={args.dt}s")
        print("   ✓ GPU-parallel vertex updates via graph coloring")
        print("   ✓ No global linear system assembly")
        print("   ✓ Near-linear scaling with mesh size")
        print("   ✓ Based on Chen et al. 2024 (SIGGRAPH)")
        print("   ✓ Stable Neo-Hookean material model")
    elif args.implicit_fem:
        print("\n💡 Fully Implicit FEM solver features:")
        print(f"   ✓ Using dt={args.dt}s")
        print("   ✓ FEM stiffness included in system matrix")
        print("   ✓ Best stability for stiff FEM materials")
        print("   ✓ Tangent stiffness K_fem = ∂f_fem/∂x from Neo-Hookean")
        print("   ✓ Matrix rebuilt every step (nonlinear accuracy)")
    elif args.implicit:
        print("\n💡 Implicit solver features:")
        print(f"   ✓ Using dt={args.dt}s")
        print("   ✓ Better stability for stiff systems")
        print("   ✓ Sparse matrix solver (BiCGStab)")
        print("   ✓ Useful for: stiff springs, constrained systems")
    else:
        print("\n💡 Semi-implicit solver features:")
        print(f"   ✓ Using dt={args.dt}s")
        print("   ✓ Fast per-step computation")
        print("   ✓ Good for moderate spring stiffness")
    
    # Run simulation
    n_steps = args.steps
    
    import time
    physics_times = []
    render_times = []
    
    for step in range(n_steps):
        # Apply zero action (let physics evolve naturally)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        
        # Time physics step
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        t1 = time.perf_counter()
        physics_times.append((t1 - t0) * 1000)
        
        # Time rendering
        if not args.no_render:
            t2 = time.perf_counter()
            env.render()
            t3 = time.perf_counter()
            render_times.append((t3 - t2) * 1000)
        
        # Print progress with timing
        if (step + 1) % 100 == 0:
            avg_physics = np.mean(physics_times[-100:])
            avg_render = np.mean(render_times[-100:]) if render_times else 0
            total_time = avg_physics + avg_render
            max_fps = 1000 / total_time if total_time > 0 else 0
            
            # Get center-of-mass velocity to detect drift
            com_vel = info.get('center_of_mass_velocity', np.array([0.0, 0.0]))
            com_speed = np.linalg.norm(com_vel)
            
            print(f"Step {step + 1}/{n_steps} | Time: {info['time']:.2f}s | Energy: {info['total_energy']:.4f} | CoM drift: {com_speed:.6f} m/s")
            print(f"  ⏱️  Physics: {avg_physics:.2f}ms | Render: {avg_render:.2f}ms | Total: {total_time:.2f}ms ({max_fps:.1f} FPS)")
            physics_times = []
            render_times = []
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print(f"Simulated time: {info['time']:.2f}s")
    print(f"Final energy: {info['total_energy']:.4f}")
    if args.vbd:
        print("✓ VBD solver completed successfully!")
    elif args.implicit_fem:
        print("✓ Fully Implicit FEM solver completed successfully!")
    elif args.implicit:
        print("✓ Implicit solver completed successfully!")
    print("=" * 60)
    
    # Keep window open for a moment if rendering
    if not args.no_render:
        print("\n⏸️  Press Ctrl+C or close window to exit...")
        try:
            import pygame
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                pygame.time.wait(100)
        except KeyboardInterrupt:
            pass
    
    env.close()


if __name__ == "__main__":
    main()
