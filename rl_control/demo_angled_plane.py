#!/usr/bin/env python3
"""
Demo: Soft Robot Locomotion on 20° Angled Plane

Tests soft robot locomotion on a uniformly tilted plane at 20 degrees.
Unlike demo_slant.py which has a flat starting section, this creates
a plane that is tilted from the very beginning.

Usage:
    python demo_angled_plane.py
    python demo_angled_plane.py --angle 30
    python demo_angled_plane.py --frequency 3.0 --force-scale 25.0

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import numpy as np
from demo_base import DemoBase, DemoConfig
from world_map import WorldMap


def create_angled_plane_terrain(
    world_width: float = 10.0,
    world_height: float = 2.5,
    angle_degrees: float = 20.0,
    resolution: float = 50.0,
    ground_thickness: float = 0.15,
) -> WorldMap:
    """
    Create a uniformly angled plane terrain.
    
    Unlike slant terrain, this has NO flat start section - the entire
    ground surface is tilted at the specified angle from x=0.
    
    Args:
        world_width: Total width of the world in units
        world_height: Total height of the world in units
        angle_degrees: Plane angle in degrees (default: 20°)
        resolution: Pixels per world unit
        ground_thickness: Thickness of ground below surface
        
    Returns:
        WorldMap object ready for simulation
    """
    width_px = int(world_width * resolution)
    height_px = int(world_height * resolution)
    
    bitmap = np.ones((height_px, width_px), dtype=np.float32)
    
    angle_rad = np.radians(angle_degrees)
    ground_thickness_px = int(ground_thickness * resolution)
    
    # Create tilted plane from x=0 (no flat section)
    for x in range(width_px):
        x_world = x / resolution
        ground_y = ground_thickness_px + int(x_world * np.tan(angle_rad) * resolution)
        
        ground_y = min(ground_y, height_px - 1)
        bitmap[:ground_y, x] = 0.0
    
    return WorldMap(bitmap=bitmap, resolution=resolution, origin=(0.0, 0.0))


class AngledPlaneDemo(DemoBase):
    """Locomotion demo on a uniformly angled plane (20° default)."""
    
    def __init__(self, angle: float = 20.0, config: DemoConfig = None):
        """
        Initialize angled plane demo.
        
        Args:
            angle: Plane angle in degrees (default: 20)
            config: Demo configuration
        """
        super().__init__(config)
        self.angle = angle
        
        # Track climb metrics
        self.max_climb = 0.0
        self.last_x = 0.0
        self.speed_samples = []
        self.avg_speed = 0.0
        self.contact_count = 0  # Debug: particles in contact with ground
        
        # Adjust physics for angled climbing
        if self.config:
            # IMPORTANT: Keep CPG direction HORIZONTAL for proper wave pattern!
            # The CPG wave must travel left→right to push the robot forward.
            self.config.direction = (1.0, 0.0)  # Horizontal wave
            
            # Anisotropic friction is ON by default. For steep slopes, use higher backward friction.
            angle_rad = math.radians(angle)
            self.config.mu_backward = 1.5   # Higher friction for slopes
            
            # Friction direction follows the SLOPE (detects sliding down correctly)
            self.config.friction_direction = (math.cos(angle_rad), math.sin(angle_rad))
            
            # Keep normal gravity - anisotropic friction handles the slope
            self.config.gravity = -0.5
    
    def create_terrain(self) -> WorldMap:
        """Create angled plane terrain."""
        return create_angled_plane_terrain(
            world_width=self.world_width,
            world_height=self.config.boxsize,
            angle_degrees=self.angle,
            resolution=50.0,
            ground_thickness=0.15,
        )
    
    def _position_robot_on_slope(self) -> None:
        """Position robot ON the sloped surface (not at fixed Y)."""
        import warp as wp
        
        cfg = self.config
        angle_rad = math.radians(self.angle)
        ground_thickness = 0.15
        
        # Get initial positions
        positions = self.env.state_in.particle_q.numpy()
        
        # Center X at start_x
        positions[:, 0] = positions[:, 0] - np.mean(positions[:, 0]) + cfg.start_x
        
        # For each particle, compute ground height at its X position
        # and position it slightly above
        min_y = np.min(positions[:, 1])
        for i in range(len(positions)):
            x = positions[i, 0]
            ground_y_at_x = ground_thickness + x * math.tan(angle_rad)
            # Position particle relative to its original height in the grid,
            # but shifted to be above the slope
            particle_height_in_grid = positions[i, 1] - min_y
            positions[i, 1] = ground_y_at_x + particle_height_in_grid + 0.02
        
        # Update state
        new_pos_wp = wp.array(positions, dtype=wp.vec2, 
                               device=self.env.state_in.particle_q.device)
        self.env.state_in.particle_q = new_pos_wp
        self.env.state_out.particle_q = wp.clone(new_pos_wp)
        
        # Update initial tracking
        self.initial_centroid_x = np.mean(positions[:, 0])
        self.initial_centroid_y = np.mean(positions[:, 1])
    
    def setup(self) -> None:
        """Setup with slope-aware robot positioning."""
        # Call parent setup
        super().setup()
        
        # Re-position robot on the slope (overrides default flat positioning)
        self._position_robot_on_slope()
        
        print(f"  Robot positioned on {self.angle}° slope at x={self.config.start_x}")
    
    def get_demo_name(self) -> str:
        return f"ANGLED PLANE - {self.angle}°"
    
    def on_reset(self) -> None:
        """Reset tracking and reposition on slope."""
        # Position robot on the slope surface
        self._position_robot_on_slope()
        
        # Update initial position tracking
        positions = self.env.state_in.particle_q.numpy()
        self.initial_centroid_x = np.mean(positions[:, 0])
        self.initial_centroid_y = np.mean(positions[:, 1])
        
        self.max_climb = 0.0
        self.last_x = self.initial_centroid_x
        self.speed_samples = []
        self.avg_speed = 0.0
    
    def on_step(self, positions: np.ndarray, cpg_output: np.ndarray) -> None:
        """Track climbing progress and speed."""
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        
        # Track max climb height
        climb = cy - self.initial_centroid_y
        if climb > self.max_climb:
            self.max_climb = climb
        
        # Track speed (along slope direction)
        dx = cx - self.last_x
        speed = dx / self.config.dt
        self.last_x = cx
        
        self.speed_samples.append(speed)
        if len(self.speed_samples) > 100:
            self.speed_samples.pop(0)
        self.avg_speed = np.mean(self.speed_samples)
        
        # Debug: count particles in contact with ground (friction zone = 0.03 units)
        if self.terrain is not None and self.frame_count % 50 == 0:
            in_contact = 0
            friction_zone = 0.03  # Must match kernel friction_zone
            for i in range(len(positions)):
                px = (positions[i, 0] - self.sdf_origin_x) * self.sdf_resolution
                py = (positions[i, 1] - self.sdf_origin_y) * self.sdf_resolution
                if 0 <= px < self.sdf_width and 0 <= py < self.sdf_height:
                    sdf_val = self.terrain.sdf[int(py), int(px)] / self.sdf_resolution
                    if sdf_val < friction_zone:  # In friction zone
                        in_contact += 1
            self.contact_count = in_contact
    
    def get_info_lines(self) -> list:
        """Add angle-specific info."""
        lines = super().get_info_lines()
        
        # Current climb height
        positions = self.env.state_in.particle_q.numpy()
        cy = np.mean(positions[:, 1])
        climb = cy - self.initial_centroid_y
        
        lines.insert(3, (f"Climb: {climb:+.3f}m (max: {self.max_climb:.3f}m)", (0, 150, 100)))
        lines.insert(4, (f"Speed: {self.avg_speed:.4f} m/s", (0, 100, 200)))
        lines.insert(5, (f"Angle: {self.angle}°", (100, 100, 100)))
        
        # Debug: show contact count
        total_particles = self.config.rows * self.config.cols
        contact_color = (0, 180, 0) if self.contact_count > 0 else (180, 0, 0)
        lines.append((f"Ground contact: {self.contact_count}/{total_particles}", contact_color))
        
        return lines
    
    def get_summary(self) -> dict:
        """Add climb-specific summary."""
        summary = super().get_summary()
        
        dx = summary['total_displacement_x']
        dy = summary['total_displacement_y']
        
        summary['max_climb'] = self.max_climb
        summary['angle'] = self.angle
        
        if self.t > 0:
            summary['average_speed'] = dx / self.t
        
        # Expected climb based on X displacement and angle
        expected_climb = dx * math.tan(math.radians(self.angle))
        
        if dx > 0.1 and dy > expected_climb * 0.5:
            summary['success'] = True
            summary['message'] = f"Robot climbed {dy:.3f}m on {self.angle}° plane!"
        elif dx > 0.05:
            summary['success'] = False
            summary['message'] = f"Robot moved but slipped (climbed {dy:.3f}m, expected {expected_climb:.3f}m)"
        else:
            summary['success'] = False
            summary['message'] = "Robot didn't make significant progress"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Angled Plane Locomotion Demo (20° default)')
    parser.add_argument('--angle', type=float, default=20.0,
                        help='Plane angle in degrees (default: 20)')
    parser.add_argument('--no-anisotropic', action='store_true',
                        help='Disable anisotropic friction (use old ratchet mode) for comparison')
    parser.add_argument('--mu-forward', type=float, default=0.15,
                        help='Forward friction coefficient (default: 0.15)')
    parser.add_argument('--mu-backward', type=float, default=1.5,
                        help='Backward friction coefficient (default: 1.5)')
    DemoBase.add_common_args(parser)
    
    args = parser.parse_args()
    config = DemoBase.config_from_args(args)
    
    # Apply friction settings from command line
    if args.no_anisotropic:
        config.anisotropic_friction = False
        print("\n⚠️  ANISOTROPIC FRICTION DISABLED - using old ratchet mode")
        print("   The robot will likely slide backward on the slope!\n")
    else:
        config.mu_forward = args.mu_forward
        config.mu_backward = args.mu_backward
    
    demo = AngledPlaneDemo(angle=args.angle, config=config)
    summary = demo.run()
    
    if 'message' in summary:
        status = "✓" if summary.get('success', False) else "✗"
        print(f"  {status} {summary['message']}")
    
    if 'average_speed' in summary:
        print(f"  Average speed: {summary['average_speed']:.4f} m/s")
    
    print(f"  Max climb achieved: {summary.get('max_climb', 0):.3f}m")


if __name__ == "__main__":
    main()

