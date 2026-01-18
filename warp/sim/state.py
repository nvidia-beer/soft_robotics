# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# State class for 2D spring-mass simulations

class State:
    """
    Represents the time-varying state of a 2D simulation.
    
    Contains particle positions, velocities, and forces.
    
    Attributes:
        particle_q: Positions (vec2), shape [particle_count]
        particle_qd: Velocities (vec2), shape [particle_count]
        particle_f: Forces (vec2), shape [particle_count]
    """
    
    def __init__(self):
        self.particle_q = None    # Positions (vec2)
        self.particle_qd = None   # Velocities (vec2)
        self.particle_f = None    # Forces (vec2)
