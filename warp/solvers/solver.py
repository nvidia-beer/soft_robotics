# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Base solver class for 2D spring mass simulations
# Adapted from newton/newton/_src/solvers/solver.py

import numpy as np
import warp as wp


class SolverBase:
    """
    Generic base class for 2D solvers.
    
    Provides helper methods for particle integration and defines the
    interface that concrete solvers must implement.
    
    Features:
        - Adaptive strain normalization for visualization
        - Common utilities for all solver types
    
    Adapted from newton.solvers.SolverBase for 2D spring-mass systems.
    """
    
    def __init__(self, model):
        """
        Initialize the solver with a model.
        
        Args:
            model: The 2D Model object containing system description
        """
        self.model = model
        
        # Adaptive strain normalization parameters (shared by all solvers)
        self._strain_update_counter = 0
        self._strain_update_interval = 10  # Update every N steps
        self._ema_alpha = 0.1  # Exponential moving average smoothing factor
    
    @property
    def device(self):
        """
        Get the device used by the solver.
        
        Returns:
            The device used by the solver
        """
        return self.model.device
    
    def step(self, state_in, state_out, dt: float):
        """
        Simulate the model for a given time step.
        
        Must be implemented by concrete solver subclasses.
        
        Args:
            state_in: The input state
            state_out: The output state
            dt: The time step (in seconds)
        """
        raise NotImplementedError("Concrete solvers must implement step()")
    
    def _update_strain_normalization(self):
        """
        Update adaptive strain normalization scale using observed strain distribution.
        
        Mathematical Approach:
        ----------------------
        We use the 95th percentile of absolute strain values as the normalization scale:
        
            ε_scale = percentile(|ε|, 95)
        
        This is then updated using an exponential moving average (EMA):
        
            ε_scale(t+1) = α * ε_scale_observed + (1-α) * ε_scale(t)
        
        where α = 0.1 is the smoothing factor.
        
        Rationale:
        ----------
        1. **Percentile vs Maximum**: Using the 95th percentile instead of maximum
           provides robustness against outliers and transient spikes.
        
        2. **Absolute Values**: We normalize based on strain magnitude, treating
           compression and tension symmetrically: |ε| ∈ [0, ε_scale]
        
        3. **EMA Smoothing**: The exponential moving average prevents rapid
           oscillations in the visualization scale while still adapting to
           changing dynamics.
        
        4. **Physical Meaning**: The resulting ε_scale represents the "typical"
           maximum strain in the current simulation state, which naturally
           reflects the system's stiffness, forcing, and dynamics.
        
        Theory:
        -------
        For a linear spring with stiffness k and rest length L₀:
        
            F = k * ε * L₀    where ε = (L - L₀) / L₀
        
        The strain ε is dimensionless and represents relative deformation.
        Typical engineering strains before yield:
            - Steel: ~0.2% (0.002)
            - Rubber: ~30% (0.3)
            - Biological tissue: ~20% (0.2)
        
        By adaptively computing ε_scale from simulation data, we obtain
        a normalization that reflects the actual material behavior and
        loading conditions, without requiring material property inputs.
        
        Implementation:
        ---------------
        - Transfers strain data from GPU to CPU for percentile computation
        - Updates model.spring_strain_scale (GPU array) with smoothed value
        - Called every 10 time steps to balance accuracy and performance
        """
        model = self.model
        
        # Transfer strains from GPU to CPU
        strains_np = model.spring_strains.numpy()
        
        # Compute 95th percentile of absolute strains
        abs_strains = np.abs(strains_np)
        if len(abs_strains) > 0:
            percentile_95 = np.percentile(abs_strains, 95)
            
            # Avoid degenerate case (minimum physical threshold)
            if percentile_95 < 1e-8:
                percentile_95 = 0.01  # Fallback minimum scale (~1% strain)
            
            # Get current scale value
            current_scale = model.spring_strain_scale.numpy()[0]
            
            # Exponential moving average update
            # This provides smooth temporal filtering while adapting to new dynamics
            new_scale = self._ema_alpha * percentile_95 + (1 - self._ema_alpha) * current_scale
            
            # Update GPU array
            model.spring_strain_scale.assign([new_scale])
    
    def _update_fem_strain_normalization(self):
        """
        Update adaptive FEM strain normalization scale using observed strain distribution.
        
        Identical algorithm to spring normalization but applied to FEM triangle strains.
        Uses 95th percentile with exponential moving average for robust, adaptive scaling.
        
        Mathematical Approach:
        ----------------------
        ε_scale_fem = percentile(|ε_fem|, 95)
        ε_scale_fem(t+1) = α * observed + (1-α) * ε_scale_fem(t)
        
        where α = 0.1 is the EMA smoothing factor.
        """
        model = self.model
        
        # Transfer FEM strains from GPU to CPU
        strains_np = model.tri_strains.numpy()
        
        # Compute 95th percentile of absolute strains
        abs_strains = np.abs(strains_np)
        if len(abs_strains) > 0:
            percentile_95 = np.percentile(abs_strains, 95)
            
            # Avoid degenerate case
            if percentile_95 < 1e-8:
                percentile_95 = 0.01  # Fallback minimum scale
            
            # Get current scale value
            current_scale = model.fem_strain_scale.numpy()[0]
            
            # Exponential moving average update
            new_scale = self._ema_alpha * percentile_95 + (1 - self._ema_alpha) * current_scale
            
            # Update GPU array
            model.fem_strain_scale.assign([new_scale])
    
    def _update_and_normalize_strains(self, update_fem=False):
        """
        Update adaptive strain normalization scales.
        
        This method updates the normalization scales periodically based on
        observed strain distribution. The actual normalization is now performed
        directly in the force evaluation kernels for efficiency (fused computation).
        
        Adaptive Scale Update:
        ----------------------
        Computes the 95th percentile of observed strain magnitudes and
        updates the normalization scale using exponential moving average.
        This provides a dynamic reference that adapts to simulation dynamics.
        
        Physical Significance:
        ----------------------
        The normalized strains (computed in force kernels) are dimensionless
        quantities in [-1, 1] range that represent relative strain magnitude
        compared to the typical operating range. They can be used for:
        
        - Visualization (color mapping)
        - Material failure detection (|ε_norm| > 1 indicates extreme strain)
        - Adaptive time stepping (reduce dt when strains are high)
        - System stability monitoring
        - Control algorithms and reinforcement learning features
        
        Note: This method only updates the scales. The actual normalization
        (ε_norm = clamp(ε / ε_scale, -1, 1)) is fused into force kernels.
        
        Args:
            update_fem: If True, update FEM strain scales (only if solver evaluates FEM forces)
        
        This is shared logic across all solvers and should be called
        at the end of each solver's step() method.
        """
        model = self.model
        
        # Update adaptive strain normalization scales periodically
        self._strain_update_counter += 1
        if self._strain_update_counter >= self._strain_update_interval:
            # Always update spring scales if springs exist
            if model.spring_count > 0:
                self._update_strain_normalization()  # Updates spring_strain_scale
            
            # Only update FEM scales if solver actually computes FEM forces
            if update_fem and hasattr(model, 'tri_count') and model.tri_count > 0:
                self._update_fem_strain_normalization()  # Updates fem_strain_scale
            
            self._strain_update_counter = 0
        
        # Note: Strain normalization is now performed in-kernel during force evaluation
        # This eliminates separate GPU kernel launches for better performance

