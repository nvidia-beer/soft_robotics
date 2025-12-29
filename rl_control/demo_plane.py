#!/usr/bin/env python3
"""
Demo: Classic Soft Robot Locomotion on Flat Ground

Basic locomotion demo - robot moves right on a flat plane.
This is the simplest test case, equivalent to rl_locomotion/demo_simple_cpg.py
but using the new modular rl_control framework with SDF collision.

Usage:
    python demo_plane.py
    python demo_plane.py --frequency 5.0
    python demo_plane.py --direction -1 0  # Move left

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from demo_base import DemoBase, DemoConfig
from world_map import WorldMap, create_plane_terrain


class PlaneDemo(DemoBase):
    """Classic locomotion on flat ground."""
    
    def __init__(self, config: DemoConfig = None):
        """
        Initialize plane demo.
        
        Args:
            config: Demo configuration
        """
        super().__init__(config)
        
        # Track velocity for speed measurement
        self.last_x = 0.0
        self.avg_speed = 0.0
        self.speed_samples = []
    
    def create_terrain(self) -> WorldMap:
        """Create flat ground terrain."""
        return create_plane_terrain(
            world_width=self.world_width,
            world_height=self.config.boxsize,
            resolution=50.0,
            ground_thickness=0.15,
        )
    
    def get_demo_name(self) -> str:
        dir_x, dir_y = self.config.direction
        if dir_x > 0:
            direction = "→ Right"
        elif dir_x < 0:
            direction = "← Left"
        elif dir_y > 0:
            direction = "↑ Up"
        else:
            direction = "↓ Down"
        return f"PLANE DEMO - {direction}"
    
    def on_reset(self) -> None:
        """Reset speed tracking."""
        self.last_x = self.initial_centroid_x
        self.avg_speed = 0.0
        self.speed_samples = []
    
    def on_step(self, positions: np.ndarray, cpg_output: np.ndarray) -> None:
        """Track locomotion speed."""
        cx = np.mean(positions[:, 0])
        
        # Calculate instantaneous speed
        dx = cx - self.last_x
        speed = dx / self.config.dt
        self.last_x = cx
        
        # Running average
        self.speed_samples.append(speed)
        if len(self.speed_samples) > 100:
            self.speed_samples.pop(0)
        self.avg_speed = np.mean(self.speed_samples)
    
    def get_info_lines(self) -> list:
        """Add speed info to HUD."""
        lines = super().get_info_lines()
        
        # Add speed information
        lines.insert(3, (f"Speed: {self.avg_speed:.4f} m/s", (0, 100, 200)))
        
        # Distance traveled
        positions = self.env.state_in.particle_q.numpy()
        cx = np.mean(positions[:, 0])
        distance = cx - self.initial_centroid_x
        lines.insert(4, (f"Distance: {distance:.3f}m", (0, 100, 0)))
        
        return lines
    
    def get_summary(self) -> dict:
        """Add locomotion-specific summary."""
        summary = super().get_summary()
        
        dx = summary['total_displacement_x']
        dir_x = self.config.direction[0]
        
        # Calculate average speed
        if self.t > 0:
            avg_speed = dx / self.t
            summary['average_speed'] = avg_speed
        
        # Check if moved in intended direction
        if dir_x > 0:
            if dx > 0.1:
                summary['success'] = True
                summary['message'] = f"Robot moved {dx:.3f}m to the right!"
            else:
                summary['success'] = False
                summary['message'] = "Robot didn't move significantly to the right"
        elif dir_x < 0:
            if dx < -0.1:
                summary['success'] = True
                summary['message'] = f"Robot moved {-dx:.3f}m to the left!"
            else:
                summary['success'] = False
                summary['message'] = "Robot didn't move significantly to the left"
        else:
            summary['success'] = True
            summary['message'] = f"Robot moved {abs(dx):.3f}m horizontally"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Plane Locomotion Demo (Classic)')
    parser.add_argument('--direction', type=float, nargs=2, default=[1.0, 0.0],
                        help='Movement direction as X Y (default: 1 0 = right)')
    DemoBase.add_common_args(parser)
    
    args = parser.parse_args()
    config = DemoBase.config_from_args(args)
    config.direction = tuple(args.direction)
    
    demo = PlaneDemo(config=config)
    summary = demo.run()
    
    if 'message' in summary:
        status = "✓" if summary.get('success', False) else "✗"
        print(f"  {status} {summary['message']}")
    
    if 'average_speed' in summary:
        print(f"  Average speed: {summary['average_speed']:.4f} m/s")


if __name__ == "__main__":
    main()

