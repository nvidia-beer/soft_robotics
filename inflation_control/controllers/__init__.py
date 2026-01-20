"""
Controllers for Inflation Control.

Volume-based PID control for soft body inflation.

Structure:
    controllers/
    ├── __init__.py          # BaseController + exports
    ├── pid/
    │   └── __init__.py      # PID controller
    ├── nengo/
    │   └── __init__.py      # Nengo SNN controller
    └── stress/
        └── __init__.py      # Stress-based controller

Usage:
    from controllers import PID
    
    pid = PID(Kp=50.0, Ki=5.0, Kd=20.0, dt=0.01)
    
    # In simulation loop:
    current_volume = model.compute_current_volume(state)
    target_volume = model.initial_volume * target_ratio
    pressure = pid.compute(target_volume, current_volume)
    forces = model.get_inflation_forces_from_pressure(state, pressure)
    solver.step(state, state_next, dt, external_forces=forces)
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for inflation controllers.
    
    Controllers compute pressure from volume error.
    """
    
    def __init__(self, dt: float, u_max: float = 10.0, **kwargs):
        """
        Initialize controller.
        
        Args:
            dt: Time step (seconds)
            u_max: Maximum pressure output
        """
        self.dt = dt
        self.u_max = u_max
        self.step_count = 0
    
    @abstractmethod
    def compute(self, target_volume: float, current_volume: float) -> float:
        """
        Compute pressure from volume error.
        
        Args:
            target_volume: Target volume (area in 2D)
            current_volume: Current volume (area in 2D)
        
        Returns:
            Pressure value (positive = inflate, negative = deflate)
        """
        pass
    
    def reset(self):
        """Reset controller state."""
        self.step_count = 0


# Import controllers
from .pid import PID

# Try to import Nengo controllers
try:
    from .nengo import NengoPID, SNN_PID_Controller
    NENGO_AVAILABLE = True
except ImportError:
    NENGO_AVAILABLE = False
    NengoPID = None
    SNN_PID_Controller = None

# Try to import Stress controllers (requires Nengo for spiking versions)
try:
    from .stress import Stress, NengoStress, SNN_Stress_Controller
    STRESS_AVAILABLE = True
except ImportError:
    STRESS_AVAILABLE = False
    Stress = None
    NengoStress = None
    SNN_Stress_Controller = None

__all__ = [
    'BaseController',
    'PID',
    'NengoPID',
    'SNN_PID_Controller',
    'NENGO_AVAILABLE',
    'Stress',
    'NengoStress',
    'SNN_Stress_Controller',
    'STRESS_AVAILABLE',
]
