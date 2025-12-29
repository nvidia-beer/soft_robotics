"""
Pygame Renderer for Spring Mass System.

This module provides shared rendering functionality used across:
- openai-gym/spring_mass_env.py
- rl_locomotion/demo_simple_cpg.py, demo_snn_gui.py  
- trajectory_tracking/tracking_env.py

Main classes:
- Renderer: Common pygame-based rendering class with all visualization features
"""

from .renderer import Renderer

__all__ = ['Renderer']
