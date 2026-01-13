# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Running Loop for Reduced-Order Simulation

Runs the soft body simulation using the reduced-order solver for real-time performance.

Usage:
    from soft_model import RunningLoop
    
    runner = RunningLoop(model, reduced_data_dir='reduced_data/')
    runner.run(duration=10.0, dt=0.01, render=True)
"""

import numpy as np
import os
import sys
import time
from typing import Optional, Any

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_order_reduction'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pygame_renderer'))


class RunningLoop:
    """
    Running loop for reduced-order soft body simulation.
    
    Uses SolverReduced for real-time simulation with pre-computed POD basis.
    
    Example:
        >>> from soft_model import RunningLoop
        >>> from warp.sim import Model
        >>> 
        >>> # Create model (same topology as training)
        >>> model = Model.from_grid(rows=10, cols=10)
        >>> 
        >>> # Create runner with pre-computed reduced data
        >>> runner = RunningLoop(model, reduced_data_dir='reduced_data/')
        >>> 
        >>> # Run simulation
        >>> runner.run(duration=10.0, dt=0.01, render=True)
    """
    
    def __init__(
        self,
        model: Any,
        reduced_data_dir: str,
        verbose: bool = True,
    ):
        """
        Initialize running loop.
        
        Args:
            model: Warp Model object (must match training topology)
            reduced_data_dir: Directory containing reduced model files
            verbose: Print progress information
        """
        import warp as wp
        
        self.model = model
        self.verbose = verbose
        
        # Load reduced model
        from warp.reduction import load_reduced_model, SolverReduced
        
        if verbose:
            print("=" * 60)
            print("RunningLoop: Loading Reduced Model")
            print("=" * 60)
        
        reduced_data = load_reduced_model(reduced_data_dir)
        
        # Create reduced solver
        self.solver = SolverReduced(
            model=model,
            reduced_data=reduced_data,
            mass=1.0,
            use_hyperreduction='rid' in reduced_data,
        )
        
        # Store compression stats
        self.stats = self.solver.get_compression_stats()
        
        if verbose:
            print(f"\n  Compression: {self.stats['compression_ratio']:.1f}x")
            print(f"  Expected speedup: ~{self.stats['linear_solve_speedup']:.0f}x")
            print("=" * 60)
    
    def run(
        self,
        duration: float = 10.0,
        dt: float = 0.01,
        render: bool = True,
        window_width: int = 1000,
        window_height: int = 600,
        fps: int = 60,
        show_info: bool = True,
    ):
        """
        Run reduced-order simulation.
        
        Simulation uses gravity-only (no external forces), same as test_renderer.py.
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step
            render: Show pygame visualization
            window_width: Render window width
            window_height: Render window height
            fps: Target FPS for rendering
            show_info: Display info overlay
        """
        import warp as wp
        
        model = self.model
        solver = self.solver
        
        n_steps = int(duration / dt)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Running Reduced-Order Simulation")
            print("=" * 60)
            print(f"  Duration: {duration}s ({n_steps} steps)")
            print(f"  dt: {dt}")
        
        # Initialize states
        state_in = model.state()
        state_out = model.state()
        
        # Setup rendering
        renderer = None
        clock = None
        screen = None
        font = None
        
        if render:
            try:
                import pygame
                from pygame_renderer import Renderer
                
                pygame.init()
                screen = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption("MOR Simulation - Reduced Order")
                clock = pygame.time.Clock()
                font = pygame.font.SysFont('monospace', 16)
                
                renderer = Renderer(
                    window_width=window_width,
                    window_height=window_height,
                    boxsize=model.boxsize,
                )
                
                if self.verbose:
                    print(f"  Render: {window_width}x{window_height} @ {fps} FPS")
            except ImportError as e:
                print(f"  Warning: pygame_renderer not available ({e}), rendering disabled")
                render = False
        
        # Performance tracking
        sim_times = []
        render_times = []
        
        # Simulation loop
        start_time = time.time()
        running = True
        step = 0
        sim_time = 0.0
        
        while running and step < n_steps:
            if render:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.key == pygame.K_r:
                            # Reset
                            state_in = model.state()
                            state_out = model.state()
                            step = 0
                            sim_time = 0.0
                
                if not running:
                    break
            
            # Step simulation (gravity-only, no external forces)
            step_start = time.time()
            solver.step(state_in, state_out, dt)
            wp.synchronize()
            sim_times.append(time.time() - step_start)
            
            # Render
            if render and renderer is not None:
                render_start = time.time()
                
                canvas = renderer.create_canvas()
                renderer.draw_grid(canvas)
                
                # Get data for rendering
                pos_np = state_out.particle_q.numpy()
                
                # Draw springs
                if model.spring_count > 0:
                    spring_indices = model.spring_indices.numpy()
                    strains = model.spring_strains_normalized.numpy() if hasattr(model, 'spring_strains_normalized') else None
                    renderer.draw_springs(canvas, spring_indices, pos_np, strains)
                
                # Draw FEM triangles if available
                if hasattr(model, 'tri_count') and model.tri_count > 0:
                    tri_indices = model.tri_indices.numpy()
                    tri_strains = model.tri_strains_normalized.numpy() if hasattr(model, 'tri_strains_normalized') else None
                    renderer.draw_fem_triangles(canvas, tri_indices, pos_np, tri_strains)
                
                # Draw particles
                renderer.draw_particles(canvas, pos_np)
                
                # Draw info overlay
                if show_info:
                    avg_sim_time = np.mean(sim_times[-100:]) * 1000 if sim_times else 0
                    real_fps = 1.0 / (np.mean(sim_times[-100:]) + 1e-6) if sim_times else 0
                    
                    info_lines = [
                        f"Time: {sim_time:.2f}s / {duration:.1f}s",
                        f"Step: {step}/{n_steps}",
                        f"Modes: {self.stats['n_reduced']} / {self.stats['n_full']} DOF",
                        f"Compression: {self.stats['compression_ratio']:.1f}x",
                        f"Sim time: {avg_sim_time:.2f}ms",
                        f"Potential FPS: {real_fps:.0f}",
                        "",
                        "Press ESC to exit",
                        "Press R to reset",
                    ]
                    
                    renderer.draw_info_text(canvas, [
                        (line, (255, 255, 255)) for line in info_lines
                    ], position=(10, 10), line_spacing=20)
                
                screen.blit(canvas, (0, 0))
                pygame.display.flip()
                clock.tick(fps)
                
                render_times.append(time.time() - render_start)
            
            # Swap states
            state_in, state_out = state_out, state_in
            
            step += 1
            sim_time += dt
        
        # Cleanup
        if render:
            pygame.quit()
        
        # Print statistics
        elapsed = time.time() - start_time
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Simulation Complete")
            print("=" * 60)
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Steps: {step}")
            print(f"  Avg sim time: {np.mean(sim_times)*1000:.2f}ms")
            print(f"  Potential FPS: {1.0/np.mean(sim_times):.0f}")
            if render_times:
                print(f"  Avg render time: {np.mean(render_times)*1000:.2f}ms")
            print("=" * 60)
        
        return {
            'elapsed': elapsed,
            'steps': step,
            'avg_sim_time_ms': np.mean(sim_times) * 1000,
            'potential_fps': 1.0 / np.mean(sim_times) if sim_times else 0,
        }
    
    def benchmark(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        warmup_steps: int = 100,
    ) -> dict:
        """
        Benchmark reduced-order solver performance.
        
        Args:
            n_steps: Number of simulation steps
            dt: Time step
            warmup_steps: Steps to skip for timing (warmup)
            
        Returns:
            Dictionary with benchmark results
        """
        import warp as wp
        
        model = self.model
        solver = self.solver
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Benchmarking Reduced-Order Solver")
            print("=" * 60)
        
        # Initialize states
        state_in = model.state()
        state_out = model.state()
        
        # Warmup
        for _ in range(warmup_steps):
            solver.step(state_in, state_out, dt)
            state_in, state_out = state_out, state_in
        
        wp.synchronize()
        
        # Benchmark
        times = []
        start = time.time()
        
        for _ in range(n_steps):
            step_start = time.time()
            solver.step(state_in, state_out, dt)
            wp.synchronize()
            times.append(time.time() - step_start)
            state_in, state_out = state_out, state_in
        
        elapsed = time.time() - start
        
        results = {
            'n_steps': n_steps,
            'total_time': elapsed,
            'avg_step_time_ms': np.mean(times) * 1000,
            'std_step_time_ms': np.std(times) * 1000,
            'min_step_time_ms': np.min(times) * 1000,
            'max_step_time_ms': np.max(times) * 1000,
            'potential_fps': 1.0 / np.mean(times),
            'compression': self.stats['compression_ratio'],
            'n_modes': self.stats['n_reduced'],
            'n_full': self.stats['n_full'],
        }
        
        if self.verbose:
            print(f"  Steps: {n_steps}")
            print(f"  Avg step time: {results['avg_step_time_ms']:.3f} ± {results['std_step_time_ms']:.3f} ms")
            print(f"  Potential FPS: {results['potential_fps']:.0f}")
            print(f"  Compression: {results['compression']:.1f}x")
            print("=" * 60)
        
        return results

