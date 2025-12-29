"""
RL Control - Force Injection & CPG Library for Soft Robotics

This package provides force injection and CPG capabilities for locomotion:

Force Injection:
- InjectForces: Basic force injection with group-based control
- InjectForcesLocomotion: Locomotion-optimized radial force modes

CPG (Central Pattern Generators):
- HopfCPG / CPG: Rate-coded Hopf oscillator (simple, fast)
- NengoCPG: Spiking neural network CPG with PES learning
- SNN_CPG_Controller: For Nengo GUI integration

Author: NBEL
License: Apache-2.0
"""

from .inject_forces_class import InjectForces
from .inject_forces_locomotion import (
    InjectForcesLocomotion,
    ForceMode,
    get_available_modes,
    describe_mode,
)
from .cpg import HopfCPG, CPG
from .snn.cpg import NengoCPG, SNN_CPG_Controller, build_snn_cpg_network

__all__ = [
    # Force injection
    'InjectForces',
    'InjectForcesLocomotion',
    'ForceMode',
    'get_available_modes',
    'describe_mode',
    # Rate-coded CPG
    'HopfCPG',
    'CPG',
    # Spiking CPG
    'NengoCPG',
    'SNN_CPG_Controller',
    'build_snn_cpg_network',
]
