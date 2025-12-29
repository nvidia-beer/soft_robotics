"""
Tessellation module for generating mesh data.

Provides refined Delaunay tessellation for creating spring-mass models
from bitmap images.

Usage:
    from tessellation import refined_delaunay_tessellation
    
    result, stats = refined_delaunay_tessellation(
        image_path='input.bmp',
        output_json='output.json',
        output_viz='output_viz.png',
    )

Output is used by:
    - Model.from_tessellation() in warp/sim/model.py
    - openai-gym/demo.py with --tessellation flag
"""

from .refined_delaunay import refined_delaunay_tessellation

__all__ = ["refined_delaunay_tessellation"]

