# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Utility Functions for Model Order Reduction

Provides convenience functions for loading/saving reduced models
and creating Warp arrays from reduced basis data.
"""

import numpy as np
import warp as wp
import os
import json
from typing import Dict, Any, Optional


def load_reduced_model(directory: str, name: str = "reduced_basis") -> Dict[str, Any]:
    """
    Load a reduced model from directory.
    
    Expected files:
    - {name}_modes.npy: POD basis matrix (n_dof, n_modes)
    - {name}_rest.npy: Rest position (n_dof,) or (n_particles, 2)
    - {name}_info.json: Metadata
    - {name}_rid.npy: (Optional) Reduced Integration Domain
    - {name}_weights.npy: (Optional) ECSW weights
    
    Args:
        directory: Path to directory containing reduced model files
        name: Base name for files (default: "reduced_basis")
        
    Returns:
        Dictionary with:
        - 'modes': Basis matrix (n_dof, n_modes)
        - 'rest_position': Rest configuration (n_particles, 2)
        - 'n_modes': Number of modes
        - 'n_particles': Number of particles
        - 'rid': (Optional) RID indices
        - 'weights': (Optional) ECSW weights
    """
    # Load modes
    modes_path = os.path.join(directory, f"{name}_modes.npy")
    if not os.path.exists(modes_path):
        raise FileNotFoundError(f"Modes file not found: {modes_path}")
    modes = np.load(modes_path)
    
    # Load rest position
    rest_path = os.path.join(directory, f"{name}_rest.npy")
    if not os.path.exists(rest_path):
        raise FileNotFoundError(f"Rest position file not found: {rest_path}")
    rest_position = np.load(rest_path)
    
    # Reshape rest position if needed
    if rest_position.ndim == 1:
        rest_position = rest_position.reshape(-1, 2)
    
    n_dof = modes.shape[0]
    n_modes = modes.shape[1]
    n_particles = n_dof // 2
    
    # Load metadata if available
    info_path = os.path.join(directory, f"{name}_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        n_modes = info.get('n_modes', n_modes)
        n_particles = info.get('n_particles', n_particles)
    
    result = {
        'modes': modes,
        'rest_position': rest_position,
        'n_modes': n_modes,
        'n_particles': n_particles,
    }
    
    # Load hyperreduction data if available
    rid_path = os.path.join(directory, f"{name}_rid.npy")
    if os.path.exists(rid_path):
        result['rid'] = np.load(rid_path)
    
    weights_path = os.path.join(directory, f"{name}_weights.npy")
    if os.path.exists(weights_path):
        result['weights'] = np.load(weights_path)
    
    print(f"Loaded reduced model from {directory}:")
    print(f"  Modes: {modes.shape}")
    print(f"  DOF: {n_dof} → {n_modes}")
    print(f"  Particles: {n_particles}")
    if 'rid' in result:
        print(f"  RID: {len(result['rid'])} elements")
    
    return result


def save_reduced_model(
    directory: str,
    modes: np.ndarray,
    rest_position: np.ndarray,
    n_particles: int,
    name: str = "reduced_basis",
    rid: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    singular_values: Optional[np.ndarray] = None,
):
    """
    Save a reduced model to directory.
    
    Args:
        directory: Output directory
        modes: Basis matrix (n_dof, n_modes)
        rest_position: Rest configuration
        n_particles: Number of particles
        name: Base name for files
        rid: (Optional) RID indices
        weights: (Optional) ECSW weights
        singular_values: (Optional) SVD singular values
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(directory, f"{name}_modes.npy"), modes)
    np.save(os.path.join(directory, f"{name}_rest.npy"), rest_position)
    
    if singular_values is not None:
        np.save(os.path.join(directory, f"{name}_sv.npy"), singular_values)
    
    if rid is not None:
        np.save(os.path.join(directory, f"{name}_rid.npy"), rid)
    
    if weights is not None:
        np.save(os.path.join(directory, f"{name}_weights.npy"), weights)
    
    # Save metadata
    info = {
        'n_full': modes.shape[0],
        'n_modes': modes.shape[1],
        'n_particles': n_particles,
        'compression_ratio': modes.shape[0] / modes.shape[1],
    }
    if rid is not None:
        info['rid_size'] = len(rid)
    
    with open(os.path.join(directory, f"{name}_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Saved reduced model to {directory}/{name}_*")


def create_reduced_arrays(
    modes: np.ndarray,
    rest_position: np.ndarray,
    device: str = 'cuda',
) -> Dict[str, wp.array]:
    """
    Create Warp arrays from NumPy reduced basis data.
    
    Args:
        modes: Basis matrix (n_dof, n_modes)
        rest_position: Rest configuration (n_particles, 2) or (n_dof,)
        device: Warp device
        
    Returns:
        Dictionary with Warp arrays:
        - 'modes': wp.array2d(dtype=float)
        - 'rest_positions': wp.array(dtype=wp.vec2)
        - 'q_r': wp.array(dtype=float) - reduced positions buffer
        - 'v_r': wp.array(dtype=float) - reduced velocities buffer
        - 'f_r': wp.array(dtype=float) - reduced forces buffer
    """
    n_dof, n_modes = modes.shape
    n_particles = n_dof // 2
    
    # Reshape rest position if needed
    if rest_position.ndim == 1:
        rest_pos_2d = rest_position.reshape(-1, 2)
    else:
        rest_pos_2d = rest_position
    
    return {
        'modes': wp.array2d(modes.astype(np.float32), dtype=float, device=device),
        'rest_positions': wp.array(rest_pos_2d.astype(np.float32), dtype=wp.vec2, device=device),
        'q_r': wp.zeros(n_modes, dtype=float, device=device),
        'v_r': wp.zeros(n_modes, dtype=float, device=device),
        'f_r': wp.zeros(n_modes, dtype=float, device=device),
    }


def compute_projection_error(
    positions: np.ndarray,
    rest_position: np.ndarray,
    modes: np.ndarray,
) -> float:
    """
    Compute projection error for given positions.
    
    error = ||x - V @ V^T @ (x - x0) - x0|| / ||x - x0||
    
    Args:
        positions: Current positions (n_dof,) or (n_particles, 2)
        rest_position: Rest configuration
        modes: Basis matrix (n_dof, n_modes)
        
    Returns:
        Relative projection error
    """
    x = positions.flatten()
    x0 = rest_position.flatten()
    
    # Displacement
    dx = x - x0
    
    # Project and reconstruct
    q_r = modes.T @ dx
    dx_reconstructed = modes @ q_r
    
    # Error
    error = np.linalg.norm(dx - dx_reconstructed)
    norm = np.linalg.norm(dx)
    
    return error / norm if norm > 1e-10 else 0.0

