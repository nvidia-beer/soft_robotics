# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Soft Model - Training and Running Model Order Reduction

This module provides:
1. TrainingLoop - Collect snapshots from full-order simulation
2. RunningLoop - Run reduced-order simulation
3. Demo integration with pygame_renderer

Based on:
- ModelOrderReduction plugin for SOFA (Goury & Duriez)
- Warp implicit FEM solver

Usage:
    # Training Phase
    from soft_model import TrainingLoop
    
    trainer = TrainingLoop(model, solver, output_dir='reduced_data/')
    trainer.collect_snapshots(n_steps=100)
    trainer.compute_basis(tolerance=1e-3)
    trainer.save()
    
    # Running Phase
    from soft_model import RunningLoop
    
    runner = RunningLoop(model, reduced_data_dir='reduced_data/')
    runner.run(duration=10.0, dt=0.01, render=True)
"""

from .training_loop import TrainingLoop
from .running_loop import RunningLoop

__all__ = ['TrainingLoop', 'RunningLoop']

