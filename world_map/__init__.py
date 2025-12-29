"""
World Map module for bitmap-based SDF collision detection.

Uses bitmap images where:
    - White (255) = passable area
    - Black (0) = walls/obstacles

Computes Signed Distance Field (SDF) for collision detection.

Includes terrain generators for locomotion testing:
    - create_plane_terrain: Flat ground (basic locomotion)
    - create_slant_terrain: Inclined plane
    - create_tunnel_terrain: Narrow passage
    - create_boulder_terrain: Obstacle to climb
    - create_combined_challenge_terrain: All obstacles in sequence

For demos with rendering, use pygame_renderer:
    cd ../pygame_renderer && ./run.sh
"""

from .world_map import WorldMap
from .terrain_generators import (
    create_plane_terrain,
    create_slant_terrain,
    create_tunnel_terrain,
    create_boulder_terrain,
    create_combined_challenge_terrain,
)

__all__ = [
    "WorldMap",
    "create_plane_terrain",
    "create_slant_terrain",
    "create_tunnel_terrain",
    "create_boulder_terrain",
    "create_combined_challenge_terrain",
]
