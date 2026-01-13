# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Proper Orthogonal Decomposition (POD) for Model Order Reduction

Based on:
- ModelOrderReduction/python/mor/reduction/script/ReadStateFilesAndComputeModes.py
- Goury & Duriez, "Fast, generic and reliable control and simulation of soft robots"

The POD algorithm computes an optimal low-dimensional basis from simulation snapshots
using Singular Value Decomposition (SVD).

Mathematical Foundation:
-----------------------
Given N snapshots of dimension n (positions minus rest position):
    X = [x₁ - x₀, x₂ - x₀, ..., xₙ - x₀]  ∈ ℝ^(n×N)

SVD decomposition:
    X = U Σ Vᵀ
    
where:
    U ∈ ℝ^(n×n)  : Left singular vectors (POD modes)
    Σ ∈ ℝ^(n×N)  : Singular values (diagonal)
    V ∈ ℝ^(N×N)  : Right singular vectors (time coefficients)

Mode Selection:
    Keep r modes such that:
    √(Σᵢ₌ᵣ σᵢ² / Σᵢ₌₁ σᵢ²) < tolerance

The reduced basis V_r = U[:, :r] captures the dominant deformation patterns.

For 2D systems with n particles:
    Full DOF: n_full = 2n (x, y for each particle)
    Reduced DOF: r (typically 20-50)
    Speedup: (n_full/r)² for linear solve
"""

import numpy as np
from typing import Any, Optional, Tuple, Union
import os


class PODReducer:
    """
    Proper Orthogonal Decomposition for computing reduced basis.
    
    Computes an optimal low-dimensional basis from simulation snapshots.
    The basis captures the dominant deformation modes with minimal information loss.
    
    Example:
        >>> reducer = PODReducer(tolerance=1e-3)
        >>> snapshots = np.random.randn(1000, 500)  # 500 snapshots of 1000 DOF
        >>> rest_pos = np.zeros(1000)
        >>> basis, info = reducer.fit(snapshots, rest_pos)
        >>> print(f"Reduced from {basis.n_full} to {basis.n_modes} modes")
    """
    
    def __init__(
        self,
        tolerance: float = 1e-3,
        max_modes: Optional[int] = None,
        add_rigid_body_modes: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the POD reducer.
        
        Args:
            tolerance: Energy tolerance for mode selection.
                       Lower = more modes, higher accuracy.
                       Typical: 1e-3 (99.9% energy) to 1e-2 (99% energy)
            max_modes: Maximum number of modes to keep (optional cap)
            add_rigid_body_modes: Tuple (rx, ry) of 0/1 for adding translation modes.
                                  (1, 1) adds both X and Y translation.
                                  For 2D: (1, 1) adds 2 rigid body modes
            verbose: Print progress information
        """
        self.tolerance = tolerance
        self.max_modes = max_modes
        self.add_rigid_body_modes = add_rigid_body_modes
        self.verbose = verbose
        
        # Computed quantities
        self.singular_values = None
        self.energy_captured = None
        self.n_modes = None
        
    def fit(
        self,
        snapshots: np.ndarray,
        rest_position: np.ndarray,
    ) -> Tuple[Any, dict]:
        """
        Compute POD basis from snapshots.
        
        Args:
            snapshots: Snapshot matrix, shape (n_snapshots, n_dof) or (n_dof, n_snapshots)
                       For 2D: n_dof = 2 * n_particles (flattened [x0,y0,x1,y1,...])
            rest_position: Rest configuration, shape (n_dof,)
        
        Returns:
            basis: ReducedBasis object containing the modes
            info: Dictionary with SVD information:
                  - singular_values: All singular values
                  - energy_ratio: Energy captured by each mode
                  - n_modes: Number of modes selected
                  - energy_captured: Total energy captured
        """
        from .reduced_basis import ReducedBasis
        
        if self.verbose:
            print("=" * 60)
            print("POD Reduction: Computing Reduced Basis")
            print("=" * 60)
        
        # Ensure rest_position is 1D
        rest_position = rest_position.flatten()
        expected_n_dof = len(rest_position)
        
        if self.verbose:
            print(f"  Input snapshots shape: {snapshots.shape}")
            print(f"  Expected DOF (from rest_position): {expected_n_dof}")
        
        # Determine correct orientation using rest_position length
        # We need (n_dof, n_snapshots) format where n_dof matches rest_position
        if snapshots.shape[0] == expected_n_dof:
            # Already (n_dof, n_snapshots)
            pass
        elif snapshots.shape[1] == expected_n_dof:
            # Need to transpose from (n_snapshots, n_dof) to (n_dof, n_snapshots)
            snapshots = snapshots.T
            if self.verbose:
                print(f"  Transposed snapshots to ({snapshots.shape[0]}, {snapshots.shape[1]})")
        else:
            raise ValueError(
                f"Snapshots shape {snapshots.shape} doesn't match expected DOF {expected_n_dof}. "
                f"Expected shape (n_dof, n_snapshots) or (n_snapshots, n_dof)."
            )
        
        n_dof, n_snapshots = snapshots.shape
        
        if self.verbose:
            print(f"  DOF: {n_dof}, Snapshots: {n_snapshots}")
        
        # Compute displacement snapshots (subtract rest position)
        snapshot_diff = np.zeros_like(snapshots)
        for i in range(n_snapshots):
            snapshot_diff[:, i] = snapshots[:, i] - rest_position
        
        # Check for NaN values (indicates unstable simulation)
        if np.any(np.isnan(snapshot_diff)):
            raise ValueError("NaN detected in snapshots! Simulation may have diverged.")
        
        # Handle rigid body modes (for 2D)
        rigid_modes = None
        rigid_mode_indices = []
        
        if self.add_rigid_body_modes is not None:
            n_particles = n_dof // 2  # 2D assumption
            translation_modes = np.zeros((n_dof, 2))  # X and Y translation
            
            # X translation mode (normalized)
            for i in range(n_particles):
                translation_modes[2 * i, 0] = 1.0 / np.sqrt(n_particles)
            
            # Y translation mode (normalized)
            for i in range(n_particles):
                translation_modes[2 * i + 1, 1] = 1.0 / np.sqrt(n_particles)
            
            # Project out rigid body modes if requested
            if self.add_rigid_body_modes[0] == 1:
                # Remove X translation component
                for j in range(n_snapshots):
                    proj = np.dot(snapshot_diff[:, j], translation_modes[:, 0])
                    snapshot_diff[:, j] -= proj * translation_modes[:, 0]
                rigid_mode_indices.append(0)
            
            if self.add_rigid_body_modes[1] == 1:
                # Remove Y translation component
                for j in range(n_snapshots):
                    proj = np.dot(snapshot_diff[:, j], translation_modes[:, 1])
                    snapshot_diff[:, j] -= proj * translation_modes[:, 1]
                rigid_mode_indices.append(1)
            
            rigid_modes = translation_modes[:, rigid_mode_indices]
            if self.verbose:
                print(f"  Added {len(rigid_mode_indices)} rigid body mode(s)")
        
        # Perform SVD (using economy SVD for efficiency)
        if self.verbose:
            print("  Computing SVD...")
        
        U, s, Vt = np.linalg.svd(snapshot_diff, full_matrices=False)
        
        self.singular_values = s
        
        # Compute energy ratios
        s_squared = s ** 2
        total_energy = np.sum(s_squared)
        
        # Determine number of modes based on tolerance
        # Find smallest r such that remaining energy / total < tolerance²
        n_modes = 1
        remaining_energy_ratio = np.sqrt(np.sum(s_squared[n_modes:]) / total_energy)
        
        while remaining_energy_ratio > self.tolerance and n_modes < len(s):
            n_modes += 1
            remaining_energy_ratio = np.sqrt(np.sum(s_squared[n_modes:]) / total_energy)
        
        # Apply max_modes cap if specified
        if self.max_modes is not None:
            n_modes = min(n_modes, self.max_modes)
        
        self.n_modes = n_modes
        self.energy_captured = 1.0 - remaining_energy_ratio ** 2
        
        if self.verbose:
            print(f"  Singular values: {len(s)}")
            print(f"  Tolerance: {self.tolerance}")
            print(f"  Selected modes: {n_modes}")
            print(f"  Energy captured: {self.energy_captured * 100:.4f}%")
        
        # Extract basis modes
        modes = U[:, :n_modes]
        
        # Prepend rigid body modes if requested
        if rigid_modes is not None and len(rigid_mode_indices) > 0:
            modes = np.concatenate([rigid_modes, modes], axis=1)
            n_modes += len(rigid_mode_indices)
            if self.verbose:
                print(f"  Total modes (with rigid body): {n_modes}")
        
        # Create ReducedBasis object
        basis = ReducedBasis(
            modes=modes,
            rest_position=rest_position,
            singular_values=s[:n_modes],
            n_particles=n_dof // 2,  # 2D assumption
        )
        
        # Info dictionary
        info = {
            'singular_values': s,
            'energy_ratio': s_squared / total_energy,
            'n_modes': n_modes,
            'energy_captured': self.energy_captured,
            'tolerance': self.tolerance,
        }
        
        if self.verbose:
            print("=" * 60)
            print(f"POD Complete: {n_dof} DOF → {n_modes} modes")
            print("=" * 60)
        
        return basis, info
    
    def save_singular_values(self, filepath: str):
        """Save singular values to file for analysis."""
        if self.singular_values is None:
            raise ValueError("Must call fit() first")
        np.savetxt(filepath, self.singular_values)
        
    @staticmethod
    def load_modes_from_file(filepath: str) -> Tuple[np.ndarray, int, int]:
        """
        Load modes from SOFA-compatible file format.
        
        File format (header: n_dof n_modes):
            500 30
            0.123 0.456 ...
            ...
        
        Args:
            filepath: Path to modes file
            
        Returns:
            modes: Mode matrix (n_dof, n_modes)
            n_dof: Number of DOF
            n_modes: Number of modes
        """
        with open(filepath, 'r') as f:
            header = f.readline().strip().split()
            n_dof, n_modes = int(header[0]), int(header[1])
            modes = np.loadtxt(f)
        
        if modes.shape[0] != n_dof:
            modes = modes.T
        
        return modes[:, :n_modes], n_dof, n_modes

