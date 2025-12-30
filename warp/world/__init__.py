"""
World module for environment/collision handling in Warp.

Provides GPU-accelerated collision detection for arbitrary 2D geometry
using Signed Distance Fields (SDF) from bitmap images.

This module is kept general and not specific to any particular environment.
"""

from .kernels_sdf import (
    apply_sdf_boundary_2d,
    apply_sdf_boundary_with_friction_2d,
    apply_sdf_boundary_anisotropic_friction_2d,
)

__all__ = [
    "apply_sdf_boundary_2d",
    "apply_sdf_boundary_with_friction_2d",
    "apply_sdf_boundary_anisotropic_friction_2d",
]



