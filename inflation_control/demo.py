#!/usr/bin/env python3
"""
Demo script for Inflating Circle Model with PID Control

Demonstrates a circle mesh that inflates toward a maximum volume.
Volume is controlled by a PID controller that adjusts the FEM rest configuration.

Left Panel: Control plots (error, rest config, volume over time)
Right Panel: Simulation visualization

Controls:
  - UP/DOWN arrows: Increase/decrease target volume ratio
  - R: Reset to initial volume
  - SPACE: Toggle auto-inflation (oscillate between min and max)
  - +/-: Adjust PID Kp gain
  - 1-5: Set FEM Poisson ratio (0.1-0.45)
  - Q/ESC: Quit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import warp as wp
from collections import deque

# Matplotlib for control plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from models import BalloonModel
from solvers import SolverImplicitFEM, SolverVBD
from pygame_renderer import Renderer
from controllers import PID


class ControlPlotter:
    """Matplotlib-based control plot panel for PID visualization."""
    
    def __init__(self, width=400, height=800, max_time=50.0, history_len=500):
        self.width = width
        self.height = height
        self.max_time = max_time
        self.history_len = history_len
        
        # History buffers
        self.time_history = deque(maxlen=history_len)
        self.error_history = deque(maxlen=history_len)
        self.rest_config_history = deque(maxlen=history_len)
        self.volume_history = deque(maxlen=history_len)
        self.target_history = deque(maxlen=history_len)
        
        self.fig = None
        self.axes = None
        self.surface = None
        self._create_figure()
    
    def _create_figure(self):
        """Create matplotlib figure with 3 subplots."""
        fig_width = self.width / 100.0
        fig_height = self.height / 100.0
        
        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        self.fig.canvas = FigureCanvasAgg(self.fig)
        
        self.axes = []
        
        # Plot 1: Volume (target vs actual) - top
        ax1 = self.fig.add_axes([0.15, 0.70, 0.80, 0.25])
        self.axes.append(ax1)
        
        # Plot 2: Error - middle
        ax2 = self.fig.add_axes([0.15, 0.40, 0.80, 0.25])
        self.axes.append(ax2)
        
        # Plot 3: Rest Config Ratio - bottom
        ax3 = self.fig.add_axes([0.15, 0.10, 0.80, 0.25])
        self.axes.append(ax3)
    
    def update(self, time_val, error, rest_config, current_volume, target_volume):
        """Add new data point to history."""
        self.time_history.append(time_val)
        self.error_history.append(error)
        self.rest_config_history.append(rest_config)
        self.volume_history.append(current_volume)
        self.target_history.append(target_volume)
    
    def render(self):
        """Render plots and return pygame surface."""
        import pygame
        
        if len(self.time_history) < 2:
            return None
        
        t = np.array(self.time_history)
        t_max = max(t[-1], 5.0)
        t_min = max(0, t_max - self.max_time)
        
        # ===== Plot 1: Volume (target vs actual) =====
        ax = self.axes[0]
        ax.clear()
        
        target = np.array(self.target_history)
        volume = np.array(self.volume_history)
        
        ax.plot(t, target, 'k--', label='Target', linewidth=2, alpha=0.7)
        ax.plot(t, volume, 'b-', label='Actual', linewidth=2)
        ax.fill_between(t, volume, target, alpha=0.2, color='blue')
        
        ax.set_ylabel('Volume', fontsize=10)
        ax.set_xlim(t_min, t_max)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('Volume Tracking', fontsize=11, fontweight='bold')
        ax.tick_params(labelbottom=False)
        
        # ===== Plot 2: Error =====
        ax = self.axes[1]
        ax.clear()
        
        error = np.array(self.error_history)
        pos_mask = error >= 0
        neg_mask = error < 0
        
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.plot(t, error, 'r-', linewidth=2)
        ax.fill_between(t, 0, error, where=pos_mask, alpha=0.3, color='red', label='Under-inflated')
        ax.fill_between(t, 0, error, where=neg_mask, alpha=0.3, color='blue', label='Over-inflated')
        
        ax.set_ylabel('Error', fontsize=10)
        ax.set_xlim(t_min, t_max)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('Volume Error (target - actual)', fontsize=11, fontweight='bold')
        ax.tick_params(labelbottom=False)
        
        # ===== Plot 3: Pressure (rest config ratio) =====
        ax = self.axes[2]
        ax.clear()
        
        rest_config = np.array(self.rest_config_history)
        
        ax.axhline(1.0, color='gray', linestyle='-', alpha=0.5)
        ax.plot(t, rest_config, color='orange', linewidth=2)
        ax.fill_between(t, 1.0, rest_config, alpha=0.3, color='orange')
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Pressure', fontsize=10)
        ax.set_xlim(t_min, t_max)
        ax.grid(True, alpha=0.3)
        ax.set_title('Inflation Pressure', fontsize=11, fontweight='bold')
        
        # Convert to pygame surface
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))[:, :, :3]
        self.surface = pygame.surfarray.make_surface(buf.swapaxes(0, 1))
        
        return self.surface
    
    def reset(self):
        """Clear history."""
        self.time_history.clear()
        self.error_history.clear()
        self.rest_config_history.clear()
        self.volume_history.clear()
        self.target_history.clear()


def main():
    parser = argparse.ArgumentParser(description="Inflating Circle Demo")
    parser.add_argument('--vbd', action='store_true',
                       help='Use VBD solver (default: implicit FEM)')
    parser.add_argument('--radius', type=float, default=0.5,
                       help='Circle radius (default: 0.5)')
    parser.add_argument('--num-boundary', type=int, default=16,
                       help='Boundary points (default: 16)')
    parser.add_argument('--num-rings', type=int, default=2,
                       help='Interior rings (default: 2)')
    parser.add_argument('--gravity', type=float, default=0.0,
                       help='Gravity (default: 0.0, use -9.8 for Earth gravity)')
    parser.add_argument('--max-volume', type=float, default=2.0,
                       help='Max volume ratio (default: 2.0 = double)')
    parser.add_argument('--fem-E', type=float, default=2000.0,
                       help='FEM Young\'s modulus (default: 2000.0)')
    parser.add_argument('--fem-nu', type=float, default=0.45,
                       help='FEM Poisson ratio (default: 0.45, near-incompressible)')
    parser.add_argument('--spring-k', type=float, default=1000.0,
                       help='Spring stiffness (default: 1000.0)')
    # PID gains for FEM rest config scaling
    parser.add_argument('--pid-kp', type=float, default=1.0,
                       help='PID proportional gain (default: 1.0)')
    parser.add_argument('--pid-ki', type=float, default=0.5,
                       help='PID integral gain (default: 0.5)')
    parser.add_argument('--pid-kd', type=float, default=0.3,
                       help='PID derivative gain (default: 0.3)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    parser.add_argument('--steps', type=int, default=5000,
                       help='Number of steps (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Computation device')
    parser.add_argument('--no-render', action='store_true',
                       help='Run without visualization')
    args = parser.parse_args()
    
    # Initialize Warp
    wp.init()
    
    print("\n" + "="*60)
    print("  INFLATING CIRCLE DEMO")
    print("="*60)
    print(f"\nInflation method: FEM rest configuration scaling")
    print(f"Material: E={args.fem_E}, nu={args.fem_nu}, spring_k={args.spring_k}")
    print(f"Gravity: {args.gravity}")
    print(f"Max volume ratio: {args.max_volume}x")
    
    # Create model
    print("\n" + "-"*40)
    model = BalloonModel(
        radius=args.radius,
        num_boundary=args.num_boundary,
        num_rings=args.num_rings,
        max_volume_ratio=args.max_volume,
        device=args.device,
        boxsize=3.0,
        spring_stiffness=args.spring_k,
        spring_damping=5.0,
        fem_E=args.fem_E,
        fem_nu=args.fem_nu,
        fem_damping=10.0,
    )
    
    # Set gravity
    model.set_gravity((0.0, args.gravity))
    
    # Create PID controller
    print("\n" + "-"*40)
    pid = PID(
        dt=args.dt,
        Kp=args.pid_kp,
        Ki=args.pid_ki,
        Kd=args.pid_kd,
        u_max=10.0,  # Max rest config adjustment
        integral_limit=5.0,
        deadband=0.0001,
    )
    print(f"PID: Kp={args.pid_kp}, Ki={args.pid_ki}, Kd={args.pid_kd}")
    
    # Create solver
    if args.vbd:
        print("\nUsing VBD solver")
        solver = SolverVBD(
            model,
            iterations=10,
            damping_coefficient=1.0,
            density=100.0
        )
    else:
        print("\nUsing Implicit FEM solver")
        solver = SolverImplicitFEM(
            model,
            dt=args.dt,
            mass=1.0,
            preconditioner_type="diag",
            solver_type="bicgstab",
            max_iterations=50,
            tolerance=1e-4,
            rebuild_matrix_every=1,
        )
    
    # Create states
    state = model.state()
    state_next = model.state()
    
    # Rendering setup
    if not args.no_render:
        import pygame
        pygame.init()
        
        plot_width = 400
        sim_width = 600
        height = 600
        total_width = plot_width + sim_width
        
        screen = pygame.display.set_mode((total_width, height))
        pygame.display.set_caption("Inflating Circle Demo - PID Pressure Control")
        clock = pygame.time.Clock()
        
        renderer = Renderer(
            window_width=sim_width,
            window_height=height,
            boxsize=model.boxsize,
        )
        
        plotter = ControlPlotter(
            width=plot_width,
            height=height,
            max_time=30.0,
            history_len=3000
        )
        plot_update_interval = 5
        frame_count = 0
        
        spring_indices_np = model.spring_indices.numpy() if model.spring_count > 0 else None
        tri_indices_np = model.tri_indices.numpy() if model.tri_count > 0 else None
    
    # Simulation state
    target_ratio = 1.0
    sim_time = 0.0
    running = True
    step = 0
    
    print("\n" + "-"*40)
    print("Controls:")
    print("  UP/DOWN: Increase/decrease target volume")
    print("  R: Reset to initial volume & PID state")
    print("  +/-: Increase/decrease PID Kp")
    print("  1-5: Set FEM Poisson ratio (0.1-0.45)")
    print("  Q/ESC: Quit")
    print("-"*40 + "\n")
    
    while running and step < args.steps:
        # Handle events
        if not args.no_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_UP:
                        target_ratio = min(target_ratio + 0.1, args.max_volume)
                        print(f"Target volume ratio: {target_ratio:.2f}")
                    elif event.key == pygame.K_DOWN:
                        target_ratio = max(target_ratio - 0.1, 1.0)
                        print(f"Target volume ratio: {target_ratio:.2f}")
                    elif event.key == pygame.K_r:
                        target_ratio = 1.0
                        sim_time = 0.0
                        pid.reset()
                        if not args.no_render:
                            plotter.reset()
                        print("Reset to initial volume & PID state")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        pid.set_gains(Kp=pid.Kp * 1.2)
                    elif event.key == pygame.K_MINUS:
                        pid.set_gains(Kp=pid.Kp / 1.2)
                    elif event.key == pygame.K_1:
                        model.set_fem_parameters(nu=0.1)
                    elif event.key == pygame.K_2:
                        model.set_fem_parameters(nu=0.2)
                    elif event.key == pygame.K_3:
                        model.set_fem_parameters(nu=0.3)
                    elif event.key == pygame.K_4:
                        model.set_fem_parameters(nu=0.4)
                    elif event.key == pygame.K_5:
                        model.set_fem_parameters(nu=0.45)
        
        # ========== INFLATION CONTROL ==========
        current_volume = model.compute_current_volume(state)
        target_volume = model.initial_volume * target_ratio
        
        # PID computes rest config adjustment
        pid_output = pid.compute(target_volume, current_volume)
        _, rest_config = model.apply_inflation(state, pid_output, target_ratio)
        
        # Update tracking
        model.current_volume_ratio = current_volume / model.initial_volume
        model.target_volume_ratio = target_ratio
        
        # Update plotter
        error = target_volume - current_volume
        if not args.no_render:
            plotter.update(sim_time, error, rest_config, current_volume, target_volume)
        
        # Physics step
        solver.step(state, state_next, args.dt, external_forces=None)
        
        sim_time += args.dt
        state, state_next = state_next, state
        
        # Render
        if not args.no_render:
            frame_count += 1
            
            positions = state.particle_q.numpy()
            spring_strains = model.spring_strains.numpy() if model.spring_strains is not None else None
            tri_strains = model.tri_strains.numpy() if model.tri_strains is not None else None
            
            # Left panel: plots
            if frame_count % plot_update_interval == 0:
                plot_surface = plotter.render()
                if plot_surface is not None:
                    screen.blit(plot_surface, (0, 0))
            
            # Volume progress bar on left panel (under plots)
            current_normalized = model.current_volume_ratio - 1.0
            max_normalized = args.max_volume - 1.0
            target_normalized = target_ratio - 1.0
            
            # Draw on screen directly (left panel area)
            renderer.draw_progress_bar(
                screen,
                value=current_normalized,
                max_value=max_normalized,
                x=10, y=height - 25,
                width=plot_width - 20, height=18,
                target_value=target_normalized,
                label=f"Volume: {model.current_volume_ratio:.2f}x / {target_ratio:.2f}x"
            )
            
            # Right panel: simulation
            canvas = renderer.create_canvas()
            renderer.draw_grid(canvas)
            
            if tri_indices_np is not None and model.tri_count > 0:
                renderer.draw_fem_triangles(canvas, tri_indices_np, positions, tri_strains)
            
            if spring_indices_np is not None and model.spring_count > 0:
                renderer.draw_springs(canvas, spring_indices_np, positions, spring_strains)
            
            # Draw particles (pink when colliding with boundary)
            if model.particle_colliding is not None:
                particle_colors = renderer.get_collision_colors(model.particle_colliding.numpy())
                renderer.draw_particles(canvas, positions, per_particle_colors=particle_colors)
            else:
                renderer.draw_particles(canvas, positions)
            
            renderer.draw_strain_legends(canvas, spring_strains=spring_strains, fem_strains=tri_strains, show_fem=True)
            
            # Info text
            info = model.get_inflation_info(state)
            info_lines = [
                (f"Step: {step} | Time: {sim_time:.1f}s", renderer.BLACK),
                (f"Ratio: {info['current_ratio']:.3f} / {target_ratio:.3f}", renderer.BLACK),
                (f"Pressure: {rest_config:.3f}", (255, 140, 0)),  # Orange
                (f"PID: Kp={pid.Kp:.1f} Ki={pid.Ki:.2f} Kd={pid.Kd:.1f}", renderer.GREY),
                (f"FEM: E={model.fem_E}, nu={model.fem_nu:.2f}", renderer.GREY),
            ]
            renderer.draw_info_text(canvas, info_lines, position=(10, 10), line_spacing=18)
            
            screen.blit(canvas, (plot_width, 0))
            renderer.draw_separator_line(screen, plot_width, end_y=height)
            
            pygame.display.flip()
            clock.tick(60)
        
        # Print progress
        if step % 100 == 0:
            info = model.get_inflation_info(state)
            print(f"  [PID step {step}] err={error:.4f} P={pid.Kp*error:.1f} I={pid.error_integral:.1f} D={pid.Kd*pid.last_error:.1f} → pressure={rest_config:.2f}")
            print(f"Step {step:4d}: ratio={info['current_ratio']:.3f} target={target_ratio:.3f}")
        
        step += 1
    
    if not args.no_render:
        pygame.quit()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
