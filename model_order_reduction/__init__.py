# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Model Order Reduction for Soft Robot Simulation

Python implementation based on:
- SOFA ModelOrderReduction plugin (Goury & Duriez, INRIA)
- "Fast, generic and reliable control and simulation of soft robots using model order reduction"

Key Components:
1. POD (Proper Orthogonal Decomposition) - Compute reduced basis from snapshots
2. ECSW (Energy Conserving Sampling and Weighting) - Hyperreduction for force evaluation
3. ReducedBasis - Storage and management of reduced basis
4. SnapshotCollector - Collect simulation snapshots for training

Usage:
    from model_order_reduction import PODReducer, ECSWHyperreducer, SnapshotCollector
    
    # 1. Collect snapshots
    collector = SnapshotCollector(model)
    for actuation in actuations:
        collector.simulate_and_record(solver, state, actuation, n_steps=100)
    
    # 2. Compute POD basis
    reducer = PODReducer(tolerance=1e-3)
    basis = reducer.fit(collector.snapshots, collector.rest_positions)
    
    # 3. Compute ECSW for hyperreduction
    hyperreducer = ECSWHyperreducer(tolerance=0.1)
    rid, weights = hyperreducer.fit(gie_data, basis)
    
    # 4. Use reduced solver
    from warp.reduction import SolverReduced
    reduced_solver = SolverReduced(model, basis, rid, weights)
"""

from .pod import PODReducer
from .ecsw import ECSWHyperreducer
from .snapshot_collector import SnapshotCollector
from .reduced_basis import ReducedBasis

__all__ = [
    'PODReducer',
    'ECSWHyperreducer', 
    'SnapshotCollector',
    'ReducedBasis',
]

