"""
Controllers for spring-mass systems.

Base controller interface and implementations for MPC, PID, Stress, and SNN-enhanced controllers.

Structure:
    controllers/
    ├── __init__.py          # BaseController + exports
    ├── mpc/
    │   └── __init__.py      # MPC (coupled spring model)
    ├── pid/
    │   └── __init__.py      # PID (multi-group feedback)
    ├── stress/
    │   └── __init__.py      # Stress (strain-based modulation)
    └── nengo/
        ├── __init__.py
        ├── base.py          # NengoControllerBase
        ├── mpc.py           # NengoMPC
        ├── pid.py           # NengoPID (PURE NEF-PD, no learning)
        └── stress.py        # NengoStress (spiking stress-adaptive)
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """
    Abstract base class for controllers.
    
    All controllers must implement compute_control method.
    """
    
    def __init__(self, n_center: int, dt: float, u_max: float = 50.0, **kwargs):
        """
        Initialize controller.
        
        Args:
            n_center: Number of controlled elements
            dt: Time step (seconds)
            u_max: Maximum control force
        """
        self.n_center = n_center
        self.dt = dt
        self.u_max = u_max
        self.u_dim = n_center * 2  # 2D forces
        self.step_count = 0
    
    @abstractmethod
    def compute_control(self, state_dict: dict, target_func) -> np.ndarray:
        """
        Compute control forces.
        
        Args:
            state_dict: State information dict
            target_func: Target function
        
        Returns:
            Control forces array
        """
        pass
    
    def reset(self):
        """Reset controller state."""
        self.step_count = 0


# Import controllers
from .mpc import MPC
from .pid import PID
from .stress import Stress

# Try to import Nengo controllers
try:
    from .nengo import NengoMPC, NengoControllerBase, NengoPID, NengoStress
    NENGO_AVAILABLE = True
except ImportError:
    NENGO_AVAILABLE = False
    NengoMPC = None
    NengoControllerBase = None
    NengoPID = None
    NengoStress = None

__all__ = [
    'BaseController',
    'MPC',
    'PID',
    'Stress',
    'NengoMPC',
    'NengoPID',
    'NengoStress',
    'NengoControllerBase',
    'NENGO_AVAILABLE'
]
