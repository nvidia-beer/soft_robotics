#!/usr/bin/env python3
"""
RL Control: Soft Robot Locomotion Challenges

A modular framework for soft robot locomotion testing with various terrains.
Uses WorldMap from world_map module for SDF-based collision.

Architecture:
- DemoBase: Base class for all locomotion demos (extend to add new demos)
- Terrain generators in world_map module (create_slant_terrain, etc.)
- Collision handled by warp/world/kernels_sdf.py

Available Terrains (from world_map):
- Slant: Inclined plane (configurable angle)
- Tunnel: Narrow passage (configurable height)  
- Boulder: Obstacle to climb over (configurable size)

Author: NBEL
License: Apache-2.0
"""

from .demo_base import DemoBase, DemoConfig

__all__ = [
    'DemoBase',
    'DemoConfig',
]
