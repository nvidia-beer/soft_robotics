#!/usr/bin/env python3
"""
Terrain Generators for Locomotion Challenges

Procedural terrain generators that create WorldMap objects.
Useful for soft robot locomotion testing.

Available terrains:
1. Plane: Flat ground (basic locomotion)
2. Slant: Inclined plane with configurable angle
3. Tunnel: Horizontal passage with configurable height
4. Boulder: Semicircular obstacle with configurable size
5. Combined: All obstacles in sequence

Author: NBEL
License: Apache-2.0
"""

import numpy as np
from .world_map import WorldMap


def create_plane_terrain(
    world_width: float = 10.0,
    world_height: float = 2.5,
    resolution: float = 50.0,
    ground_thickness: float = 0.15,
) -> WorldMap:
    """
    Create a simple flat ground terrain for basic locomotion.
    
    Args:
        world_width: Total width of the world in units
        world_height: Total height of the world in units
        resolution: Pixels per world unit
        ground_thickness: Thickness of ground below surface
        
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    ground_thickness_px = int(ground_thickness * resolution)
    
    # Ground (floor only)
    bitmap[:ground_thickness_px, :] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))


def create_slant_terrain(
    world_width: float = 10.0,
    world_height: float = 2.5,
    angle_degrees: float = 45.0,
    resolution: float = 50.0,
    ground_thickness: float = 0.3,
    flat_start: float = 1.0,
) -> WorldMap:
    """
    Create a slanted ground terrain for uphill locomotion.
    
    The terrain has a flat starting section, then slopes upward at the specified angle.
    
    Args:
        world_width: Total width of the world in units
        world_height: Total height of the world in units
        angle_degrees: Slope angle in degrees (default: 45°)
        resolution: Pixels per world unit
        ground_thickness: Thickness of ground below surface
        flat_start: Length of flat section at start
        
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    angle_rad = np.radians(angle_degrees)
    flat_start_px = int(flat_start * resolution)
    ground_thickness_px = int(ground_thickness * resolution)
    
    for x in range(width_px):
        if x < flat_start_px:
            ground_y = ground_thickness_px
        else:
            x_from_start = (x - flat_start_px) / resolution
            ground_y = ground_thickness_px + int(x_from_start * np.tan(angle_rad) * resolution)
        
        ground_y = min(ground_y, height_px - 1)
        bitmap[:ground_y, x] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))


def create_tunnel_terrain(
    world_width: float = 10.0,
    world_height: float = 2.5,
    tunnel_height_ratio: float = 0.9,
    robot_height: float = 0.5,
    resolution: float = 50.0,
    tunnel_start: float = 2.0,
    tunnel_length: float = 4.0,
    ground_thickness: float = 0.15,
) -> WorldMap:
    """
    Create a tunnel terrain for squeeze-through locomotion.
    
    The tunnel height is a percentage of the robot height, forcing
    the soft robot to compress itself to pass through.
    
    Args:
        world_width: Total width of the world in units
        world_height: Total height of the world in units
        tunnel_height_ratio: Tunnel height as ratio of robot height (default: 0.9 = 90%)
        robot_height: Expected height of the soft robot
        resolution: Pixels per world unit
        tunnel_start: X position where tunnel begins
        tunnel_length: Length of the tunnel
        ground_thickness: Thickness of ground below
        
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    tunnel_height = robot_height * tunnel_height_ratio
    tunnel_start_px = int(tunnel_start * resolution)
    tunnel_end_px = int((tunnel_start + tunnel_length) * resolution)
    ground_thickness_px = int(ground_thickness * resolution)
    ceiling_y_px = int((ground_thickness + tunnel_height) * resolution)
    
    # Ground (floor)
    bitmap[:ground_thickness_px, :] = 0.0
    
    # Tunnel ceiling
    for x in range(tunnel_start_px, min(tunnel_end_px, width_px)):
        bitmap[ceiling_y_px:, x] = 0.0
    
    # Smooth entry/exit ramps
    ramp_length_px = int(0.3 * resolution)
    
    for i in range(ramp_length_px):
        x = tunnel_start_px - ramp_length_px + i
        if 0 <= x < width_px:
            ramp_y = ceiling_y_px + (ramp_length_px - i)
            if ramp_y < height_px:
                bitmap[ramp_y:, x] = 0.0
    
    for i in range(ramp_length_px):
        x = tunnel_end_px + i
        if 0 <= x < width_px:
            ramp_y = ceiling_y_px + i
            if ramp_y < height_px:
                bitmap[ramp_y:, x] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))


def create_boulder_terrain(
    world_width: float = 10.0,
    world_height: float = 2.5,
    boulder_size_ratio: float = 0.5,
    robot_size: float = 0.5,
    resolution: float = 50.0,
    boulder_position: float = 3.5,
    ground_thickness: float = 0.15,
) -> WorldMap:
    """
    Create a boulder terrain for climb-over locomotion.
    
    The boulder is a semicircular obstacle that the robot must climb over.
    
    Args:
        world_width: Total width of the world in units
        world_height: Total height of the world in units
        boulder_size_ratio: Boulder radius as ratio of robot size (default: 0.5 = 50%)
        robot_size: Reference size of the soft robot
        resolution: Pixels per world unit
        boulder_position: X position of boulder center
        ground_thickness: Thickness of ground below
        
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    ground_thickness_px = int(ground_thickness * resolution)
    boulder_radius = robot_size * boulder_size_ratio
    boulder_radius_px = int(boulder_radius * resolution)
    boulder_center_x_px = int(boulder_position * resolution)
    boulder_center_y_px = ground_thickness_px
    
    # Ground
    bitmap[:ground_thickness_px, :] = 0.0
    
    # Boulder (semicircle)
    for y in range(ground_thickness_px, min(ground_thickness_px + boulder_radius_px + 1, height_px)):
        for x in range(max(0, boulder_center_x_px - boulder_radius_px),
                       min(width_px, boulder_center_x_px + boulder_radius_px + 1)):
            dx = x - boulder_center_x_px
            dy = y - boulder_center_y_px
            if dx * dx + dy * dy <= boulder_radius_px * boulder_radius_px:
                bitmap[y, x] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))


def create_combined_challenge_terrain(
    world_width: float = 15.0,
    world_height: float = 2.5,
    resolution: float = 50.0,
    robot_height: float = 0.5,
    robot_size: float = 0.5,
    slant_angle: float = 30.0,
    tunnel_height_ratio: float = 0.9,
    boulder_size_ratio: float = 0.5,
) -> WorldMap:
    """
    Create a combined challenge terrain with all three obstacles.
    
    Sequence: flat start -> boulder -> tunnel -> slant
    
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    ground_thickness = 0.15
    ground_thickness_px = int(ground_thickness * resolution)
    
    boulder_start = 2.0
    boulder_end = 4.0
    tunnel_start = 5.0
    tunnel_end = 8.0
    slant_start = 9.0
    
    # Ground
    bitmap[:ground_thickness_px, :] = 0.0
    
    # Boulder
    boulder_radius = robot_size * boulder_size_ratio
    boulder_radius_px = int(boulder_radius * resolution)
    boulder_center_x_px = int((boulder_start + boulder_end) / 2 * resolution)
    boulder_center_y_px = ground_thickness_px
    
    for y in range(ground_thickness_px, min(ground_thickness_px + boulder_radius_px + 1, height_px)):
        for x in range(max(0, boulder_center_x_px - boulder_radius_px),
                       min(width_px, boulder_center_x_px + boulder_radius_px + 1)):
            dx = x - boulder_center_x_px
            dy = y - boulder_center_y_px
            if dx * dx + dy * dy <= boulder_radius_px * boulder_radius_px:
                bitmap[y, x] = 0.0
    
    # Tunnel
    tunnel_height = robot_height * tunnel_height_ratio
    ceiling_y_px = int((ground_thickness + tunnel_height) * resolution)
    tunnel_start_px = int(tunnel_start * resolution)
    tunnel_end_px = int(tunnel_end * resolution)
    
    for x in range(tunnel_start_px, min(tunnel_end_px, width_px)):
        bitmap[ceiling_y_px:, x] = 0.0
    
    # Slant
    slant_start_px = int(slant_start * resolution)
    angle_rad = np.radians(slant_angle)
    
    for x in range(slant_start_px, width_px):
        x_from_start = (x - slant_start_px) / resolution
        slant_y = ground_thickness_px + int(x_from_start * np.tan(angle_rad) * resolution)
        slant_y = min(slant_y, height_px - 1)
        bitmap[:slant_y, x] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))
