#!/usr/bin/env python3
"""
Demo: Soft Robot Locomotion Through Tunnel

Tests soft robot ability to squeeze through a narrow tunnel.
Default height: 90% of robot height (configurable via --tunnel-ratio)

Usage:
    python demo_tunnel.py
    python demo_tunnel.py --tunnel-ratio 0.8
    python demo_tunnel.py --tunnel-ratio 0.7 --frequency 3.0

Author: NBEL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pygame
from demo_base import DemoBase, DemoConfig
from world_map import WorldMap, create_tunnel_terrain


class TunnelDemo(DemoBase):
    """Locomotion demo through narrow tunnel."""
    
    def __init__(
        self,
        tunnel_ratio: float = 0.9,
        tunnel_length: float = 3.0,
        config: DemoConfig = None
    ):
        """
        Initialize tunnel demo.
        
        Args:
            tunnel_ratio: Tunnel height as ratio of robot height (0.0-1.0)
            tunnel_length: Length of tunnel in world units
            config: Demo configuration
        """
        super().__init__(config)
        self.tunnel_ratio = tunnel_ratio
        self.tunnel_length = tunnel_length
        self.tunnel_start = 2.5
        
        # State tracking
        self.entered_tunnel = False
        self.exited_tunnel = False
        self.entry_time = None
        self.exit_time = None
        
        # Adjust physics for compression
        if self.config:
            self.config.spring_coeff = 40.0  # Softer for compression
    
    def create_terrain(self) -> WorldMap:
        """Create tunnel terrain."""
        return create_tunnel_terrain(
            world_width=self.world_width,
            world_height=self.config.boxsize,
            tunnel_height_ratio=self.tunnel_ratio,
            robot_height=self.robot_height,
            resolution=50.0,
            tunnel_start=self.tunnel_start,
            tunnel_length=self.tunnel_length,
            ground_thickness=0.15,
        )
    
    def get_demo_name(self) -> str:
        return f"TUNNEL DEMO - {self.tunnel_ratio*100:.0f}% Height"
    
    def on_reset(self) -> None:
        """Reset tunnel tracking state."""
        self.entered_tunnel = False
        self.exited_tunnel = False
        self.entry_time = None
        self.exit_time = None
    
    def on_step(self, positions: np.ndarray, cpg_output: np.ndarray) -> None:
        """Track tunnel progress."""
        cx = np.mean(positions[:, 0])
        tunnel_exit_x = self.tunnel_start + self.tunnel_length
        
        if not self.entered_tunnel and cx >= self.tunnel_start:
            self.entered_tunnel = True
            self.entry_time = self.t
            print(f"  [t={self.t:.2f}s] Entered tunnel!")
        
        if self.entered_tunnel and not self.exited_tunnel and cx >= tunnel_exit_x:
            self.exited_tunnel = True
            self.exit_time = self.t
            transit = self.exit_time - self.entry_time
            print(f"  [t={self.t:.2f}s] Exited tunnel! Transit time: {transit:.2f}s")
    
    def get_info_lines(self) -> list:
        """Add tunnel-specific info."""
        lines = super().get_info_lines()
        
        # Calculate compression
        positions = self.env.state_in.particle_q.numpy()
        current_height = np.max(positions[:, 1]) - np.min(positions[:, 1])
        compression = 1.0 - (current_height / self.robot_height)
        
        status = "Exited!" if self.exited_tunnel else ("In Tunnel" if self.entered_tunnel else "Approaching")
        
        lines.insert(3, (f"Compression: {compression*100:.1f}%", (100, 0, 100)))
        lines.insert(4, (f"Status: {status}", (0, 100, 0) if self.exited_tunnel else (0, 0, 0)))
        return lines
    
    def draw_custom(self, canvas: pygame.Surface) -> None:
        """Draw tunnel entry/exit markers."""
        scale = self.config.window_height / self.config.boxsize
        
        entry_px = int(self.tunnel_start * scale)
        exit_px = int((self.tunnel_start + self.tunnel_length) * scale)
        
        pygame.draw.line(canvas, (0, 200, 0), (entry_px, 0), (entry_px, self.config.window_height), 2)
        pygame.draw.line(canvas, (200, 0, 0), (exit_px, 0), (exit_px, self.config.window_height), 2)
    
    def get_summary(self) -> dict:
        """Add tunnel-specific summary."""
        summary = super().get_summary()
        summary['tunnel_ratio'] = self.tunnel_ratio
        summary['entered_tunnel'] = self.entered_tunnel
        summary['exited_tunnel'] = self.exited_tunnel
        
        if self.exited_tunnel:
            transit = self.exit_time - self.entry_time
            summary['transit_time'] = transit
            summary['success'] = True
            summary['message'] = f"Robot passed through tunnel in {transit:.2f}s!"
        elif self.entered_tunnel:
            summary['success'] = False
            summary['message'] = "Robot entered tunnel but didn't exit"
        else:
            summary['success'] = False
            summary['message'] = "Robot didn't reach the tunnel"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Tunnel Locomotion Demo')
    parser.add_argument('--tunnel-ratio', type=float, default=0.9,
                        help='Tunnel height as ratio of robot height (default: 0.9)')
    parser.add_argument('--tunnel-length', type=float, default=3.0,
                        help='Tunnel length in world units (default: 3.0)')
    DemoBase.add_common_args(parser)
    
    args = parser.parse_args()
    config = DemoBase.config_from_args(args)
    
    demo = TunnelDemo(
        tunnel_ratio=args.tunnel_ratio,
        tunnel_length=args.tunnel_length,
        config=config
    )
    summary = demo.run()
    
    if 'message' in summary:
        status = "✓" if summary.get('success', False) else "✗"
        print(f"  {status} {summary['message']}")


if __name__ == "__main__":
    main()

