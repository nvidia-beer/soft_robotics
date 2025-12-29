"""
Nengo-based Spiking Neural Network CPG

Provides spiking neural network implementations of Central Pattern Generators:
- NengoCPG: NEF-based spiking CPG (Hopf oscillator network)
- SNN_CPG_Controller: Spiking CPG for Nengo GUI integration

The Hopf oscillator is implemented using NEF with:
- 2D ensemble per oscillator (x, y Cartesian coordinates)
- Recurrent connections for Hopf dynamics
- Coupling connections for traveling wave coordination

Example (standalone):
    >>> from rl_locomotion import NengoCPG
    >>> cpg = NengoCPG(num_groups=9, frequency=2.0, direction=1.0)
    >>> output = cpg.step()  # Get 9D output in [-1, 1]

Example (Nengo GUI):
    >>> from rl_locomotion.snn import SNN_CPG_Controller
    >>> cpg_ctrl = SNN_CPG_Controller(num_groups=9, frequency=2.0)
    >>> with model:
    >>>     cpg_ctrl.build_all(model)

Author: NBEL
License: Apache-2.0
"""

from .cpg import NengoCPG, SNN_CPG_Controller, build_snn_cpg_network

__all__ = [
    'NengoCPG',
    'SNN_CPG_Controller',
    'build_snn_cpg_network',
]

