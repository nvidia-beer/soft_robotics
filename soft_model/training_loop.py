# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Training Loop for Model Order Reduction

Collects snapshots from full-order simulation and computes the POD basis.

Training Process:
1. Run full-order simulation (gravity-only)
2. Collect position snapshots at regular intervals
3. Compute POD basis using SVD
4. (Optional) Compute ECSW hyperreduction weights
5. Save reduced model for use with SolverReduced

Reference:
- ModelOrderReduction/python/mor/reduction/reduceModel.py
"""

import numpy as np
import os
import sys
import time
from typing import Optional, Any

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'warp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_order_reduction'))


class TrainingLoop:
    """
    Training loop for Model Order Reduction.
    
    Collects snapshots from full-order simulation and computes reduced basis.
    
    Example:
        >>> from soft_model import TrainingLoop
        >>> from warp.sim import Model
        >>> from warp.solvers import SolverImplicit
        >>> 
        >>> # Create model and solver
        >>> model = Model.from_grid(rows=10, cols=10)
        >>> solver = SolverImplicit(model)
        >>> 
        >>> # Training
        >>> trainer = TrainingLoop(model, solver, output_dir='reduced_data/')
        >>> 
        >>> trainer.collect_snapshots(
        >>>     n_steps=1000,
        >>>     record_interval=5,
        >>> )
        >>> 
        >>> trainer.compute_basis(tolerance=1e-3)
        >>> trainer.save()
    """
    
    def __init__(
        self,
        model: Any,
        solver: Any,
        output_dir: str = 'reduced_data/',
        verbose: bool = True,
    ):
        """
        Initialize training loop.
        
        Args:
            model: Warp Model object
            solver: Warp Solver object (e.g., SolverImplicit, SolverImplicitFEM)
            output_dir: Directory to save reduced model data
            verbose: Print progress information
        """
        self.model = model
        self.solver = solver
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Import model order reduction components
        from model_order_reduction import SnapshotCollector, PODReducer, ECSWHyperreducer
        
        self.snapshot_collector = SnapshotCollector()
        self.pod_reducer = None
        self.ecsw_reducer = None
        
        # Results
        self.basis = None
        self.pod_info = None
        self.rid = None
        self.weights = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print("=" * 60)
            print("TrainingLoop Initialized")
            print("=" * 60)
            print(f"  Model: {model.particle_count} particles, {model.spring_count} springs")
            if hasattr(model, 'tri_count'):
                print(f"  FEM: {model.tri_count} triangles")
            print(f"  Output: {output_dir}")
            print("=" * 60)
    
    def collect_snapshots(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        record_interval: int = 5,
        render: bool = False,
        window_width: int = 800,
        window_height: int = 600,
    ):
        """
        Run full-order simulation and collect snapshots.
        
        Simulation uses gravity-only (no external forces), same as test_renderer.py.
        
        Args:
            n_steps: Total number of simulation steps
            dt: Time step
            record_interval: Record snapshot every N steps
            render: Show pygame visualization during training
            window_width: Render window width
            window_height: Render window height
        """
        import warp as wp
        
        model = self.model
        solver = self.solver
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 1: Collecting Snapshots")
            print("=" * 60)
            print(f"  Steps: {n_steps}")
            print(f"  dt: {dt}")
            print(f"  Record interval: {record_interval}")
            print(f"  Expected snapshots: ~{n_steps // record_interval}")
        
        # Initialize states
        state_in = model.state()
        state_out = model.state()
        
        # Set rest position from initial state
        rest_pos = state_in.particle_q.numpy()
        self.snapshot_collector.set_rest_position(rest_pos)
        
        # Setup rendering if requested
        renderer = None
        clock = None
        screen = None
        if render:
            try:
                import pygame
                from pygame_renderer import Renderer
                
                pygame.init()
                screen = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption("MOR Training - Snapshot Collection")
                clock = pygame.time.Clock()
                
                renderer = Renderer(
                    window_width=window_width,
                    window_height=window_height,
                    boxsize=model.boxsize,
                )
            except ImportError:
                print("  Warning: pygame_renderer not available, rendering disabled")
                render = False
        
        # Simulation loop
        start_time = time.time()
        running = True
        
        for step in range(n_steps):
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
                
                if not running:
                    break
            
            # Step simulation (gravity-only, no external forces)
            solver.step(state_in, state_out, dt)
            
            # Record snapshot
            if step % record_interval == 0:
                positions = state_out.particle_q.numpy()
                self.snapshot_collector.add_snapshot(positions)
            
            # Render
            if render and renderer is not None:
                canvas = renderer.create_canvas()
                renderer.draw_grid(canvas)
                
                # Get data for rendering
                pos_np = state_out.particle_q.numpy()
                
                # Draw springs
                if model.spring_count > 0:
                    spring_indices = model.spring_indices.numpy()
                    strains = model.spring_strains_normalized.numpy() if hasattr(model, 'spring_strains_normalized') else None
                    renderer.draw_springs(canvas, spring_indices, pos_np, strains)
                
                # Draw particles
                renderer.draw_particles(canvas, pos_np)
                
                # Draw info
                renderer.draw_info_text(canvas, [
                    (f"Step: {step}/{n_steps}", (255, 255, 255)),
                    (f"Snapshots: {self.snapshot_collector.n_snapshots}", (255, 255, 255)),
                ], position=(10, 10))
                
                screen.blit(canvas, (0, 0))
                pygame.display.flip()
                clock.tick(60)
            
            # Swap states
            state_in, state_out = state_out, state_in
            
            # Progress
            if self.verbose and step % 100 == 0:
                progress = (step + 1) / n_steps * 100
                print(f"  Progress: {progress:.1f}% ({self.snapshot_collector.n_snapshots} snapshots)")
        
        elapsed = time.time() - start_time
        
        if render:
            pygame.quit()
        
        if self.verbose:
            print(f"\n  Collected {self.snapshot_collector.n_snapshots} snapshots in {elapsed:.1f}s")
            print("=" * 60)
    
    def compute_basis(
        self,
        tolerance: float = 1e-3,
        max_modes: Optional[int] = None,
        add_rigid_body_modes: bool = False,
    ):
        """
        Compute POD basis from collected snapshots.
        
        Args:
            tolerance: Energy tolerance for mode selection (lower = more modes)
            max_modes: Maximum number of modes (optional cap)
            add_rigid_body_modes: Add translation modes (useful for locomotion)
        """
        from model_order_reduction import PODReducer
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 2: Computing POD Basis")
            print("=" * 60)
        
        # Get snapshots
        snapshots = self.snapshot_collector.get_snapshots()
        rest_position = self.snapshot_collector.rest_position
        
        # Create POD reducer
        rigid_modes = (1, 1) if add_rigid_body_modes else None
        self.pod_reducer = PODReducer(
            tolerance=tolerance,
            max_modes=max_modes,
            add_rigid_body_modes=rigid_modes,
            verbose=self.verbose,
        )
        
        # Compute basis
        self.basis, self.pod_info = self.pod_reducer.fit(snapshots, rest_position)
        
        if self.verbose:
            print(f"\n  Reduced: {self.basis.n_full} DOF → {self.basis.n_modes} modes")
            print(f"  Compression: {self.basis.compression_ratio:.1f}x")
            print(f"  Linear solve speedup: ~{self.basis.linear_solve_speedup:.0f}x")
            print("=" * 60)
    
    def compute_hyperreduction(
        self,
        gie_data: Optional[np.ndarray] = None,
        tolerance: float = 0.1,
    ):
        """
        Compute ECSW hyperreduction (optional - for further speedup).
        
        Args:
            gie_data: GIE matrix (n_samples, n_elements). If None, skipped.
            tolerance: ECSW tolerance (higher = fewer elements)
        """
        if gie_data is None:
            if self.verbose:
                print("\n  Skipping hyperreduction (no GIE data)")
            return
        
        from model_order_reduction import ECSWHyperreducer
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Phase 3: Computing ECSW Hyperreduction")
            print("=" * 60)
        
        self.ecsw_reducer = ECSWHyperreducer(
            tolerance=tolerance,
            verbose=self.verbose,
        )
        
        self.rid, self.weights = self.ecsw_reducer.fit(gie_data)
        
        if self.verbose:
            print(f"\n  RID size: {len(self.rid)} elements")
            print("=" * 60)
    
    def save(self, name: str = "reduced_basis"):
        """
        Save reduced model to output directory.
        
        Args:
            name: Base name for saved files
        """
        if self.basis is None:
            raise ValueError("Must call compute_basis() first")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Saving Reduced Model")
            print("=" * 60)
        
        # Save using ReducedBasis method
        self.basis.save(self.output_dir, name)
        
        # Save hyperreduction if computed
        if self.rid is not None:
            np.save(os.path.join(self.output_dir, f"{name}_rid.npy"), self.rid)
            np.save(os.path.join(self.output_dir, f"{name}_weights.npy"), self.weights)
        
        # Save snapshots for debugging/analysis
        self.snapshot_collector.save(os.path.join(self.output_dir, "snapshots.npz"))
        
        # Save POD info
        if self.pod_info is not None:
            import json
            with open(os.path.join(self.output_dir, f"{name}_pod_info.json"), 'w') as f:
                # Convert numpy types to Python types
                info = {k: float(v) if isinstance(v, (np.floating, np.integer)) else 
                        v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in self.pod_info.items()}
                json.dump(info, f, indent=2)
        
        if self.verbose:
            print(f"  Saved to: {self.output_dir}")
            print("=" * 60)
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        stats = {
            'n_snapshots': self.snapshot_collector.n_snapshots,
            'n_dof': self.snapshot_collector.n_dof,
        }
        
        if self.basis is not None:
            stats.update({
                'n_modes': self.basis.n_modes,
                'compression_ratio': self.basis.compression_ratio,
                'linear_solve_speedup': self.basis.linear_solve_speedup,
            })
        
        if self.pod_info is not None:
            stats['energy_captured'] = self.pod_info.get('energy_captured', None)
        
        if self.rid is not None:
            stats['rid_size'] = len(self.rid)
        
        return stats

