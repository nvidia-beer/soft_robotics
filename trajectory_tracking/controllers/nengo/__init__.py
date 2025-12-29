"""
Nengo-based spiking neural network controllers.

Provides:
- NengoPID: NEF-based spiking PD controller (standalone, PURE - no learning)
- NengoStress: NEF-based spiking stress-adaptive controller (full spiking)
- NengoMPC: MPC + SNN prediction (standalone)
- NengoCPG: NEF-based spiking CPG (Hopf oscillator network)

For Nengo GUI integration, import directly:
  from controllers.nengo.pid import SNN_PID_Controller
  from controllers.nengo.stress import SNN_Stress_Controller
  from controllers.nengo.cpg import SNN_CPG_Controller

Controllers:
- pid.py: PURE NEF-PD from Zaidel et al. (no learning, no strain)
- stress.py: Full spiking stress-adaptive control (PD + strain feedback)
- cpg.py: Hopf oscillator CPG as spiking network (locomotion patterns)

Based on:
- Zaidel et al., "Neuromorphic NEF-Based Inverse Kinematics and PID Control"
  Frontiers in Neurorobotics, 2021. https://doi.org/10.3389/fnbot.2021.631159
- Ijspeert, "Central pattern generators for locomotion control", 2008
"""

from .base import NengoControllerBase
from .cpg import NengoCPG
from .mpc import NengoMPC
from .pid import NengoPID
from .stress import NengoStress

__all__ = [
    'NengoControllerBase',
    'NengoCPG',
    'NengoMPC', 
    'NengoPID',
    'NengoStress',
]
