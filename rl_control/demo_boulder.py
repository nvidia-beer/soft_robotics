#!/usr/bin/env python3
"""
Demo: Soft Robot Climbing Over Boulder

Tests soft robot ability to climb over a semicircular obstacle.
Default size: 50% of robot size (configurable via --boulder-ratio)

Usage:
    python demo_boulder.py
    python demo_boulder.py --boulder-ratio 0.6
    python demo_boulder.py --boulder-ratio 0.4 --frequency 5.0

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pygame
from demo_base import DemoBase, DemoConfig
from world_map import WorldMap, create_boulder_terrain


class BoulderDemo(DemoBase):
    """Locomotion demo climbing over boulder."""
    
    def __init__(
        self,
        boulder_ratio: float = 0.5,
        boulder_position: float = 3.0,
        config: DemoConfig = None
    ):
        """
        Initialize boulder demo.
        
        Args:
            boulder_ratio: Boulder radius as ratio of robot size (0.0-1.0)
            boulder_position: X position of boulder center
            config: Demo configuration
        """
        super().__init__(config)
        self.boulder_ratio = boulder_ratio
        self.boulder_position = boulder_position
        
        # State tracking
        self.reached_boulder = False
        self.passed_boulder = False
        self.reach_time = None
        self.pass_time = None
        self.max_height = 0.0
        
        # Adjust physics
        if self.config:
            self.config.gravity = -0.6
            self.config.force_scale = 25.0
    
    def create_terrain(self) -> WorldMap:
        """Create boulder terrain."""
        boulder_radius = self.robot_size * self.boulder_ratio
        print(f"  Boulder: radius={boulder_radius:.3f} at x={self.boulder_position:.1f}")
        print(f"  Robot: size={self.robot_size:.3f}, starts at x={self.config.start_x:.1f}")
        return create_boulder_terrain(
            world_width=self.world_width,
            world_height=self.config.boxsize,
            boulder_size_ratio=self.boulder_ratio,
            robot_size=self.robot_size,
            resolution=50.0,
            boulder_position=self.boulder_position,
            ground_thickness=0.15,
        )
    
    def get_demo_name(self) -> str:
        return f"BOULDER DEMO - {self.boulder_ratio*100:.0f}% Size"
    
    def on_reset(self) -> None:
        """Reset boulder tracking state."""
        self.reached_boulder = False
        self.passed_boulder = False
        self.reach_time = None
        self.pass_time = None
        self.max_height = 0.0
    
    def on_step(self, positions: np.ndarray, cpg_output: np.ndarray) -> None:
        """Track boulder progress."""
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        
        boulder_radius = self.robot_size * self.boulder_ratio
        
        self.max_height = max(self.max_height, cy)
        
        if not self.reached_boulder and cx >= self.boulder_position - boulder_radius:
            self.reached_boulder = True
            self.reach_time = self.t
            print(f"  [t={self.t:.2f}s] Reached boulder!")
        
        if self.reached_boulder and not self.passed_boulder:
            if cx >= self.boulder_position + boulder_radius + 0.1:
                self.passed_boulder = True
                self.pass_time = self.t
                climb_time = self.pass_time - self.reach_time
                print(f"  [t={self.t:.2f}s] Passed boulder! Climb time: {climb_time:.2f}s")
    
    def get_info_lines(self) -> list:
        """Add boulder-specific info."""
        lines = super().get_info_lines()
        
        positions = self.env.state_in.particle_q.numpy()
        cy = np.mean(positions[:, 1])
        
        status = "Passed!" if self.passed_boulder else ("Climbing" if self.reached_boulder else "Approaching")
        
        lines.insert(3, (f"Height: {cy:.3f} (max: {self.max_height:.3f})", (100, 50, 0)))
        lines.insert(4, (f"Status: {status}", (0, 100, 0) if self.passed_boulder else (0, 0, 0)))
        return lines
    
    def draw_custom(self, canvas: pygame.Surface) -> None:
        """Draw boulder position marker."""
        scale = self.config.window_height / self.config.boxsize
        boulder_px = int(self.boulder_position * scale)
        pygame.draw.line(canvas, (150, 100, 50), (boulder_px, 0), (boulder_px, self.config.window_height), 2)
    
    def get_summary(self) -> dict:
        """Add boulder-specific summary."""
        summary = super().get_summary()
        summary['boulder_ratio'] = self.boulder_ratio
        summary['reached_boulder'] = self.reached_boulder
        summary['passed_boulder'] = self.passed_boulder
        summary['max_height'] = self.max_height
        
        if self.passed_boulder:
            climb_time = self.pass_time - self.reach_time
            summary['climb_time'] = climb_time
            summary['success'] = True
            summary['message'] = f"Robot climbed over boulder in {climb_time:.2f}s!"
        elif self.reached_boulder:
            summary['success'] = False
            summary['message'] = "Robot reached boulder but didn't pass it"
        else:
            summary['success'] = False
            summary['message'] = "Robot didn't reach the boulder"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Boulder Climbing Demo')
    parser.add_argument('--boulder-ratio', type=float, default=0.5,
                        help='Boulder size as ratio of robot size (default: 0.5)')
    parser.add_argument('--boulder-position', type=float, default=3.0,
                        help='Boulder X position (default: 3.0)')
    DemoBase.add_common_args(parser)
    
    args = parser.parse_args()
    config = DemoBase.config_from_args(args)
    
    demo = BoulderDemo(
        boulder_ratio=args.boulder_ratio,
        boulder_position=args.boulder_position,
        config=config
    )
    summary = demo.run()
    
    if 'message' in summary:
        status = "✓" if summary.get('success', False) else "✗"
        print(f"  {status} {summary['message']}")


if __name__ == "__main__":
    main()

