"""
RL Locomotion - Force Injection & CPG Library for Soft Robotics

This package provides force injection and CPG capabilities for locomotion:

Force Injection:
- BalloonForces: Balloon-style force injection for locomotion

CPG (Central Pattern Generators):
- HopfCPG / CPG: Rate-coded Hopf oscillator (simple, fast)
- NengoCPG: Spiking neural network CPG with PES learning
- SNN_CPG_Controller: For Nengo GUI integration

Author: NBEL
License: Apache-2.0
"""

from .balloon_forces import BalloonForces
from .cpg import HopfCPG, CPG
from .snn.cpg import NengoCPG, SNN_CPG_Controller, build_snn_cpg_network

__all__ = [
    # Force injection
    'BalloonForces',
    # Rate-coded CPG
    'HopfCPG',
    'CPG',
    # Spiking CPG
    'NengoCPG',
    'SNN_CPG_Controller',
    'build_snn_cpg_network',
]
