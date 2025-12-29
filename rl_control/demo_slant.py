#!/usr/bin/env python3
"""
Demo: Soft Robot Locomotion on Slanted Terrain

Tests soft robot ability to climb an inclined plane using CPG-based locomotion.
Default angle: 45 degrees (configurable via --angle)

Usage:
    python demo_slant.py
    python demo_slant.py --angle 30
    python demo_slant.py --angle 60 --frequency 5.0

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from demo_base import DemoBase, DemoConfig
from world_map import WorldMap, create_slant_terrain


class SlantDemo(DemoBase):
    """Locomotion demo on slanted terrain."""
    
    def __init__(self, angle: float = 45.0, config: DemoConfig = None):
        """
        Initialize slant demo.
        
        Args:
            angle: Slope angle in degrees
            config: Demo configuration
        """
        super().__init__(config)
        self.angle = angle
        
        # Adjust physics for climbing
        if self.config:
            self.config.gravity = -0.8  # Stronger gravity for challenge
            self.config.force_scale = 25.0
    
    def create_terrain(self) -> WorldMap:
        """Create slanted terrain."""
        return create_slant_terrain(
            world_width=self.world_width,
            world_height=self.config.boxsize,
            angle_degrees=self.angle,
            resolution=50.0,
            ground_thickness=0.15,
            flat_start=1.0,
        )
    
    def get_demo_name(self) -> str:
        return f"SLANT DEMO - {self.angle}° Incline"
    
    def get_info_lines(self) -> list:
        """Add slant-specific info."""
        lines = super().get_info_lines()
        
        # Calculate climb progress
        positions = self.env.state_in.particle_q.numpy()
        import numpy as np
        cy = np.mean(positions[:, 1])
        climb = cy - self.initial_centroid_y
        
        lines.insert(3, (f"Climb height: {climb:+.3f}m", (0, 100, 0)))
        return lines
    
    def get_summary(self) -> dict:
        """Add climb-specific summary."""
        summary = super().get_summary()
        
        dx = summary['total_displacement_x']
        dy = summary['total_displacement_y']
        
        if dx > 0.05 and dy > 0.01:
            summary['success'] = True
            summary['message'] = "Robot climbed the slope successfully!"
        elif dx > 0.05:
            summary['success'] = False
            summary['message'] = "Robot moved forward but didn't climb much"
        else:
            summary['success'] = False
            summary['message'] = "Robot didn't make significant progress"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Slant Locomotion Demo')
    parser.add_argument('--angle', type=float, default=45.0,
                        help='Slant angle in degrees (default: 45)')
    DemoBase.add_common_args(parser)
    
    args = parser.parse_args()
    config = DemoBase.config_from_args(args)
    
    demo = SlantDemo(angle=args.angle, config=config)
    summary = demo.run()
    
    if 'message' in summary:
        status = "✓" if summary.get('success', False) else "✗"
        print(f"  {status} {summary['message']}")


if __name__ == "__main__":
    main()

