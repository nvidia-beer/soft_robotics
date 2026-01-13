# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Reduced Basis Storage and Management

Container class for POD basis and related data.
Supports saving/loading and provides projection operations.
"""

import numpy as np
from typing import Optional
import os
import json


class ReducedBasis:
    """
    Container for POD reduced basis.
    
    Stores:
    - POD modes (V_r matrix)
    - Rest position
    - Singular values
    - Metadata (n_particles, n_modes, etc.)
    
    Provides projection operations:
    - project_to_reduced: x_r = V_r^T @ (x - x0)
    - project_to_full: x = V_r @ x_r + x0
    
    Example:
        >>> basis = ReducedBasis(modes, rest_position, singular_values, n_particles)
        >>> 
        >>> # Project full state to reduced
        >>> q_reduced = basis.project_to_reduced(q_full)
        >>> 
        >>> # Reconstruct full state
        >>> q_full_approx = basis.project_to_full(q_reduced)
    """
    
    def __init__(
        self,
        modes: np.ndarray,
        rest_position: np.ndarray,
        singular_values: Optional[np.ndarray] = None,
        n_particles: Optional[int] = None,
    ):
        """
        Initialize reduced basis.
        
        Args:
            modes: Mode matrix V_r, shape (n_dof, n_modes)
            rest_position: Rest configuration x0, shape (n_dof,)
            singular_values: SVD singular values, shape (n_modes,)
            n_particles: Number of particles (for 2D: n_dof = 2 * n_particles)
        """
        self.modes = modes.astype(np.float32)
        self.rest_position = rest_position.flatten().astype(np.float32)
        self.singular_values = singular_values.astype(np.float32) if singular_values is not None else None
        
        self.n_full = modes.shape[0]
        self.n_modes = modes.shape[1]
        self.n_particles = n_particles if n_particles else self.n_full // 2
        
        # Precompute V_r^T for efficient projection
        self._modes_T = self.modes.T
        
    @property
    def compression_ratio(self) -> float:
        """Compute compression ratio: n_full / n_modes."""
        return self.n_full / self.n_modes
    
    @property
    def linear_solve_speedup(self) -> float:
        """Estimated speedup for linear solve: (n_full/n_modes)^2."""
        return self.compression_ratio ** 2
    
    def project_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        """
        Project full state to reduced coordinates.
        
        q_r = V_r^T @ (x - x0)
        
        Args:
            x_full: Full state vector, shape (n_dof,) or (n_dof, n_samples)
            
        Returns:
            q_r: Reduced coordinates, shape (n_modes,) or (n_modes, n_samples)
        """
        x_full = np.atleast_2d(x_full)
        if x_full.shape[0] == 1:
            x_full = x_full.T
        
        # Subtract rest position
        x_diff = x_full - self.rest_position.reshape(-1, 1)
        
        # Project: q_r = V^T @ x_diff
        q_r = self._modes_T @ x_diff
        
        return q_r.squeeze()
    
    def project_to_full(self, q_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct full state from reduced coordinates.
        
        x = V_r @ q_r + x0
        
        Args:
            q_reduced: Reduced coordinates, shape (n_modes,) or (n_modes, n_samples)
            
        Returns:
            x_full: Reconstructed full state, shape (n_dof,) or (n_dof, n_samples)
        """
        q_r = np.atleast_2d(q_reduced)
        if q_r.shape[0] == 1:
            q_r = q_r.T
        
        # Reconstruct: x = V @ q_r + x0
        x_full = self.modes @ q_r + self.rest_position.reshape(-1, 1)
        
        return x_full.squeeze()
    
    def project_matrix_to_reduced(self, M_full: np.ndarray) -> np.ndarray:
        """
        Project a matrix to reduced space.
        
        M_r = V_r^T @ M @ V_r
        
        Args:
            M_full: Full matrix, shape (n_dof, n_dof)
            
        Returns:
            M_r: Reduced matrix, shape (n_modes, n_modes)
        """
        return self._modes_T @ M_full @ self.modes
    
    def project_vector_to_reduced(self, f_full: np.ndarray) -> np.ndarray:
        """
        Project a force vector to reduced space.
        
        f_r = V_r^T @ f
        
        Args:
            f_full: Full force vector, shape (n_dof,)
            
        Returns:
            f_r: Reduced force, shape (n_modes,)
        """
        return self._modes_T @ f_full
    
    def reconstruction_error(self, x_full: np.ndarray) -> float:
        """
        Compute reconstruction error.
        
        error = ||x - V_r @ V_r^T @ (x - x0) - x0|| / ||x - x0||
        
        Args:
            x_full: Full state to test
            
        Returns:
            Relative reconstruction error
        """
        x_diff = x_full - self.rest_position
        x_reconstructed = self.modes @ (self._modes_T @ x_diff)
        error = np.linalg.norm(x_diff - x_reconstructed)
        norm = np.linalg.norm(x_diff)
        return error / norm if norm > 1e-10 else 0.0
    
    def save(self, directory: str, name: str = "reduced_basis"):
        """
        Save reduced basis to directory.
        
        Creates:
        - {name}_modes.npy: Mode matrix
        - {name}_rest.npy: Rest position
        - {name}_info.json: Metadata
        
        Args:
            directory: Output directory
            name: Base name for files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save arrays
        np.save(os.path.join(directory, f"{name}_modes.npy"), self.modes)
        np.save(os.path.join(directory, f"{name}_rest.npy"), self.rest_position)
        
        if self.singular_values is not None:
            np.save(os.path.join(directory, f"{name}_sv.npy"), self.singular_values)
        
        # Save metadata
        info = {
            'n_full': self.n_full,
            'n_modes': self.n_modes,
            'n_particles': self.n_particles,
            'compression_ratio': self.compression_ratio,
        }
        with open(os.path.join(directory, f"{name}_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Saved reduced basis to {directory}/{name}_*")
    
    @classmethod
    def load(cls, directory: str, name: str = "reduced_basis") -> "ReducedBasis":
        """
        Load reduced basis from directory.
        
        Args:
            directory: Directory containing saved files
            name: Base name used when saving
            
        Returns:
            Loaded ReducedBasis
        """
        modes = np.load(os.path.join(directory, f"{name}_modes.npy"))
        rest_position = np.load(os.path.join(directory, f"{name}_rest.npy"))
        
        sv_path = os.path.join(directory, f"{name}_sv.npy")
        singular_values = np.load(sv_path) if os.path.exists(sv_path) else None
        
        info_path = os.path.join(directory, f"{name}_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            n_particles = info.get('n_particles')
        else:
            n_particles = None
        
        return cls(modes, rest_position, singular_values, n_particles)
    
    def __repr__(self) -> str:
        return (f"ReducedBasis(n_full={self.n_full}, n_modes={self.n_modes}, "
                f"particles={self.n_particles}, compression={self.compression_ratio:.1f}x)")

