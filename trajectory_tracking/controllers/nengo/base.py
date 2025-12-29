"""
Base class for Nengo spiking neural network controllers.

Provides common functionality for GPU acceleration and simulator management.
"""

from .. import BaseController
import numpy as np
import nengo

# Try to import nengo_dl for TensorFlow GPU acceleration
try:
    import nengo_dl
    NENGO_DL_AVAILABLE = True
except ImportError:
    NENGO_DL_AVAILABLE = False


class NengoControllerBase(BaseController):
    """
    Base class for Nengo spiking neural network controllers.
    
    Provides:
    - GPU acceleration support via nengo-dl (TensorFlow)
    - Simulator creation with CPU fallback
    - Common cleanup logic
    """
    
    def __init__(
        self,
        n_center: int,
        dt: float,
        u_max: float = 50.0,
        tau: float = 0.05,
        debug: bool = False,
        use_gpu: bool = True,
    ):
        """
        Initialize base Nengo controller.
        
        Args:
            n_center: Number of controlled elements
            dt: Time step (seconds)
            u_max: Maximum control force
            tau: Synaptic time constant in seconds
            debug: Enable debug output
            use_gpu: Enable GPU acceleration (nengo-dl)
        """
        super().__init__(n_center, dt, u_max)
        
        # Neural parameters
        self.tau = tau
        self.debug = debug
        self.use_gpu = use_gpu
        
        # GPU state (set by subclass)
        self.gpu_enabled = False
        
        # Simulators (managed by subclass)
        self._simulators = []
    
    def _create_simulator(self, model, use_gpu: bool = None):
        """
        Create a Nengo simulator with optional GPU acceleration.
        
        Priority: nengo-dl (TensorFlow) > nengo (CPU)
        
        Args:
            model: Nengo network model
            use_gpu: Override GPU setting (None = use self.use_gpu)
        
        Returns:
            tuple: (simulator, gpu_success)
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
            
        gpu_success = False
        self._backend = "cpu"
        
        # Try nengo-dl (TensorFlow)
        if use_gpu and NENGO_DL_AVAILABLE:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                device = "/gpu:0" if gpus else "/cpu:0"
                
                sim = nengo_dl.Simulator(
                    model, 
                    dt=self.dt, 
                    progress_bar=False,
                    device=device
                )
                gpu_success = bool(gpus)
                self._backend = "nengo-dl (GPU)" if gpus else "nengo-dl (CPU)"
                if self.debug:
                    print(f"  ✓ Using nengo-dl on {device}")
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️  nengo-dl failed: {e}")
                sim = None
        else:
            sim = None
        
        # Fallback: standard nengo (CPU)
        if sim is None:
            sim = nengo.Simulator(model, dt=self.dt, progress_bar=False)
            self._backend = "nengo (CPU)"
        
        self._simulators.append(sim)
        return sim, gpu_success
    
    def _reset_simulator(self, model, old_sim):
        """
        Reset a simulator, trying to maintain GPU backend.
        
        Args:
            model: Nengo network model
            old_sim: Previous simulator to close
        
        Returns:
            tuple: (new_simulator, gpu_success)
        """
        try:
            old_sim.close()
        except:
            pass
        
        # Remove from tracked list
        if old_sim in self._simulators:
            self._simulators.remove(old_sim)
        
        return self._create_simulator(model, self.gpu_enabled)
    
    def _close_all_simulators(self):
        """Close all tracked simulators."""
        for sim in self._simulators:
            try:
                sim.close()
            except:
                pass
        self._simulators.clear()
    
    def _print_gpu_status(self):
        """Print GPU backend status."""
        backend = getattr(self, '_backend', 'unknown')
        if self.gpu_enabled:
            print(f"  ⚡ Backend: {backend}")
        else:
            print(f"  Backend: {backend}")
    
    def __del__(self):
        """Clean up all simulators."""
        self._close_all_simulators()
