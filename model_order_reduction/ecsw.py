# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
"""
Energy Conserving Sampling and Weighting (ECSW) for Hyperreduction

Based on:
- ModelOrderReduction/python/mor/reduction/script/ReadGieFileAndComputeRIDandWeights.py
- Farhat et al., "Structure-preserving, stability, and accuracy properties of the ECSW method"

ECSW selects a small subset of elements (triangles/springs) that can approximate
the full force vector when evaluated with appropriate weights.

Mathematical Foundation:
-----------------------
Given internal energy gradients G ∈ ℝ^(m×n) where:
    m = number of time samples
    n = number of elements (springs/triangles)
    
Each row g_i represents the energy gradient contribution of all elements at time i.

Target: b = Σⱼ gᵢⱼ (sum over all elements = full force)

Find sparse weights ξ ∈ ℝⁿ such that:
    ||G·ξ - b|| < τ·||b||
    
where τ is the tolerance and ξ has few non-zero entries.

Algorithm (Greedy ECSW):
1. Initialize ξ = 0, RID = ∅
2. While ||G·ξ - b|| > τ·||b||:
   a. Find element j with maximum correlation: j* = argmax(Gᵀ(b - G·ξ))
   b. Add j* to RID
   c. Update ξ by least squares on selected columns
   d. Ensure non-negativity of weights
3. Return RID and weights ξ

Result:
- RID: Reduced Integration Domain (typically 5-15% of elements)
- Weights: Scaling factors for each selected element
"""

import numpy as np
from typing import Tuple, List, Optional, Set
import sys


class ECSWHyperreducer:
    """
    Energy Conserving Sampling and Weighting for hyperreduction.
    
    Selects a minimal subset of elements that can approximate the full
    internal force vector with prescribed accuracy.
    
    Example:
        >>> hyperreducer = ECSWHyperreducer(tolerance=0.1)
        >>> # gie_data: (n_samples, n_elements) energy gradient contributions
        >>> rid, weights = hyperreducer.fit(gie_data)
        >>> print(f"Selected {len(rid)} of {gie_data.shape[1]} elements ({100*len(rid)/gie_data.shape[1]:.1f}%)")
    """
    
    def __init__(
        self,
        tolerance: float = 0.1,
        verbose: bool = True,
        max_elements: Optional[int] = None,
    ):
        """
        Initialize the ECSW hyperreducer.
        
        Args:
            tolerance: Relative error tolerance (0.01 = 1%, 0.1 = 10%)
                       Lower = more elements, higher accuracy
                       Typical: 0.01 to 0.2
            verbose: Print progress information
            max_elements: Maximum number of elements to select (optional cap)
        """
        self.tolerance = tolerance
        self.verbose = verbose
        self.max_elements = max_elements
        
        # Computed quantities
        self.rid = None
        self.weights = None
        self.final_error = None
        
    def fit(
        self,
        gie_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Reduced Integration Domain and weights.
        
        Args:
            gie_data: Generalized Internal Energy gradient contributions.
                      Shape (n_samples, n_elements) where:
                      - n_samples: number of time samples
                      - n_elements: number of force-contributing elements
                      Each entry G[i,j] is the force contribution of element j at time i.
        
        Returns:
            rid: Indices of selected elements (Reduced Integration Domain)
            weights: Weights for selected elements
        """
        if self.verbose:
            print("=" * 60)
            print("ECSW Hyperreduction: Computing RID and Weights")
            print("=" * 60)
        
        # Ensure correct shape (samples x elements)
        G = gie_data.astype(np.float64)
        
        if G.shape[0] > G.shape[1]:
            # Likely transposed, but we'll keep as is
            pass
        
        n_samples, n_elements = G.shape
        
        if self.verbose:
            print(f"  Samples: {n_samples}, Elements: {n_elements}")
            print(f"  Tolerance: {self.tolerance}")
        
        # Remove zero rows (timesteps with no internal forces)
        row_sums = np.sum(np.abs(G), axis=1)
        nonzero_rows = row_sums > 1e-12
        G_clean = G[nonzero_rows, :]
        n_samples_clean = G_clean.shape[0]
        
        if n_samples_clean < n_samples:
            if self.verbose:
                print(f"  Removed {n_samples - n_samples_clean} zero rows")
        
        if n_samples_clean == 0:
            raise ValueError("All rows are zero! No valid force data.")
        
        # Compute target: sum over all elements for each sample
        # b = Σⱼ G[:,j] for each time sample
        b = np.sum(G_clean, axis=1, keepdims=True)
        
        if self.verbose:
            print(f"  Target norm: {np.linalg.norm(b):.6f}")
        
        # Run ECSW selection
        rid, xi = self._select_ecsw(G_clean, b)
        
        # Store results
        self.rid = np.array(sorted(list(rid)), dtype=np.int32)
        self.weights = np.zeros(n_elements, dtype=np.float64)
        self.weights[self.rid] = xi[self.rid].flatten()
        
        # Compute final error
        final_approx = G_clean @ self.weights.reshape(-1, 1)
        self.final_error = np.linalg.norm(final_approx - b) / np.linalg.norm(b)
        
        if self.verbose:
            print("=" * 60)
            print(f"ECSW Complete:")
            print(f"  Selected: {len(self.rid)} / {n_elements} elements ({100*len(self.rid)/n_elements:.1f}%)")
            print(f"  Final relative error: {self.final_error:.6f}")
            print("=" * 60)
        
        return self.rid, self.weights[self.rid]
    
    def _select_ecsw(
        self,
        G: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[Set[int], np.ndarray]:
        """
        Core ECSW greedy selection algorithm.
        
        Based on ModelOrderReduction/python/mor/reduction/script/ReadGieFileAndComputeRIDandWeights.py
        
        Args:
            G: Clean GIE matrix (n_samples, n_elements)
            b: Target vector (n_samples, 1)
            
        Returns:
            rid: Set of selected element indices
            xi: Weight vector (n_elements, 1)
        """
        n_samples, n_elements = G.shape
        
        # Initialize
        rid = set()
        xi = np.zeros((n_elements, 1), dtype=np.float64)
        
        # Target error threshold
        target_error = self.tolerance * np.linalg.norm(b)
        current_error = np.linalg.norm(G @ xi - b)
        initial_margin = current_error - target_error
        
        iteration = 0
        max_iterations = n_elements  # Safety limit
        
        while current_error > target_error and iteration < max_iterations:
            if self.max_elements is not None and len(rid) >= self.max_elements:
                if self.verbose:
                    print(f"  Reached max_elements limit: {self.max_elements}")
                break
            
            # Progress update
            if self.verbose:
                progress = (initial_margin - (current_error - target_error)) / initial_margin
                progress = max(0.0, min(1.0, progress))
                self._update_progress(progress)
            
            # Compute residual
            residual = b - G @ xi
            
            # Find element with maximum correlation to residual
            correlations = G.T @ residual
            best_idx = int(np.argmax(correlations))
            
            # Add to RID
            rid.add(best_idx)
            
            # Solve least squares on selected elements
            G_selected = G[:, list(rid)]
            eta_tilde = self._solve_least_squares(G_selected, b)
            
            # Ensure non-negativity (NNLS constraint)
            while not np.all(eta_tilde > -1e-12):
                # Find elements that would go negative
                rid_list = list(rid)
                xi_selected = xi[rid_list]
                
                # Identify negative entries
                neg_mask = (eta_tilde - xi_selected) < -1e-12
                neg_indices = np.where(neg_mask.flatten())[0]
                
                if len(neg_indices) == 0:
                    break
                
                # Find maximum feasible step
                vec1 = -xi_selected[neg_indices]
                vec2 = eta_tilde[neg_indices] - xi_selected[neg_indices]
                
                # Avoid division by zero
                safe_vec2 = np.where(np.abs(vec2) > 1e-15, vec2, np.sign(vec2) * 1e-15)
                steps = vec1 / safe_vec2
                max_step = np.min(steps[steps > 0]) if np.any(steps > 0) else 0.0
                
                # Update xi for selected elements
                xi[rid_list] = xi_selected + max_step * (eta_tilde - xi_selected)
                
                # Remove elements that hit zero
                zero_mask = np.abs(xi[rid_list]) < 1e-12
                for i, idx in enumerate(rid_list):
                    if zero_mask[i]:
                        rid.discard(idx)
                
                # Re-solve with remaining elements
                if len(rid) > 0:
                    G_selected = G[:, list(rid)]
                    eta_tilde = self._solve_least_squares(G_selected, b)
                else:
                    break
            
            # Update weights
            rid_list = list(rid)
            if len(rid_list) > 0:
                xi[rid_list] = eta_tilde.reshape(-1, 1)
            
            # Update error
            current_error = np.linalg.norm(G @ xi - b)
            iteration += 1
        
        if self.verbose:
            self._update_progress(1.0)
            print()  # Newline after progress
        
        return rid, xi
    
    @staticmethod
    def _solve_least_squares(G_selected: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve least squares: min ||G_selected @ eta - b||²
        
        Using normal equations: (Gᵀ G) η = Gᵀ b
        """
        GtG = G_selected.T @ G_selected
        Gtb = G_selected.T @ b
        
        # Add small regularization for numerical stability
        reg = 1e-10 * np.eye(GtG.shape[0])
        
        try:
            eta = np.linalg.solve(GtG + reg, Gtb)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            eta = np.linalg.lstsq(G_selected, b, rcond=None)[0]
        
        return eta
    
    @staticmethod
    def _update_progress(progress: float):
        """Display progress bar."""
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)
        sys.stdout.write(f'\r  Progress: [{bar}] {100*progress:.1f}%')
        sys.stdout.flush()
    
    def save_rid_and_weights(self, rid_filepath: str, weights_filepath: str):
        """
        Save RID and weights to files (SOFA-compatible format).
        
        Args:
            rid_filepath: Path to save RID indices
            weights_filepath: Path to save weights
        """
        if self.rid is None:
            raise ValueError("Must call fit() first")
        
        # RID file: header with count, then indices
        n_rid = len(self.rid)
        np.savetxt(rid_filepath, self.rid, header=f'{n_rid} 1', comments='', fmt='%d')
        
        # Weights file: all weights (sparse, most are zero)
        np.savetxt(weights_filepath, self.weights, header=f'{len(self.weights)} 1', comments='', fmt='%.5f')
        
    @staticmethod
    def load_rid_from_file(filepath: str) -> np.ndarray:
        """Load RID from file."""
        with open(filepath, 'r') as f:
            header = f.readline().strip().split()
            n_rid = int(header[0])
            rid = np.loadtxt(f, dtype=np.int32)
        return rid[:n_rid] if len(rid) > n_rid else rid
    
    @staticmethod
    def load_weights_from_file(filepath: str) -> np.ndarray:
        """Load weights from file."""
        with open(filepath, 'r') as f:
            header = f.readline().strip().split()
            n_weights = int(header[0])
            weights = np.loadtxt(f, dtype=np.float64)
        return weights[:n_weights] if len(weights) > n_weights else weights

