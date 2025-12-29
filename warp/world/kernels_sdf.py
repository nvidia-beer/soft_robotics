"""
Warp kernels for SDF-based collision detection.

Uses precomputed Signed Distance Fields (SDF) for efficient collision detection
with arbitrary 2D geometry. The SDF can come from any source (bitmap, procedural, etc.)

This module is general and not tied to any specific world map implementation.
"""

import warp as wp


@wp.func
def bilinear_sample(
    data: wp.array2d(dtype=float),
    px: float,
    py: float,
    width: int,
    height: int,
) -> float:
    """
    Sample 2D array with bilinear interpolation.
    
    Args:
        data: 2D array to sample
        px, py: Pixel coordinates (can be fractional)
        width, height: Array dimensions
        
    Returns:
        Interpolated value at (px, py)
    """
    # Clamp to valid range
    px = wp.clamp(px, 0.0, float(width) - 1.001)
    py = wp.clamp(py, 0.0, float(height) - 1.001)
    
    x0 = int(px)
    y0 = int(py)
    x1 = wp.min(x0 + 1, width - 1)
    y1 = wp.min(y0 + 1, height - 1)
    
    fx = px - float(x0)
    fy = py - float(y0)
    
    v00 = data[y0, x0]
    v01 = data[y0, x1]
    v10 = data[y1, x0]
    v11 = data[y1, x1]
    
    v0 = v00 * (1.0 - fx) + v01 * fx
    v1 = v10 * (1.0 - fx) + v11 * fx
    
    return v0 * (1.0 - fy) + v1 * fy


@wp.kernel
def apply_sdf_boundary_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    sdf: wp.array2d(dtype=float),
    sdf_grad_x: wp.array2d(dtype=float),
    sdf_grad_y: wp.array2d(dtype=float),
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    restitution: float,
):
    """
    Apply boundary conditions using SDF (Signed Distance Field).
    
    Collision detection and response for arbitrary 2D geometry.
    Simple reflection, no friction.
    
    Args:
        x: Particle positions [N, vec2]
        v: Particle velocities [N, vec2]
        sdf: Signed distance field (positive = inside, negative = outside/wall)
        sdf_grad_x, sdf_grad_y: SDF gradient components (for normal computation)
        resolution: Pixels per world unit
        origin_x, origin_y: World coordinates of SDF origin (bottom-left)
        width, height: SDF dimensions in pixels
        restitution: Bounce coefficient (0 = no bounce, 1 = full bounce)
    """
    tid = wp.tid()
    
    pos = x[tid]
    vel = v[tid]
    
    # Convert world position to pixel coordinates
    px = (pos[0] - origin_x) * resolution
    py = (pos[1] - origin_y) * resolution
    
    # Check if within SDF bounds
    if px < 0.0 or px >= float(width) or py < 0.0 or py >= float(height):
        # Outside SDF bounds - clamp to boundary
        world_w = float(width) / resolution
        world_h = float(height) / resolution
        
        if pos[0] < origin_x:
            pos = wp.vec2(origin_x, pos[1])
            vel = wp.vec2(wp.abs(vel[0]) * restitution, vel[1])
        elif pos[0] > origin_x + world_w:
            pos = wp.vec2(origin_x + world_w, pos[1])
            vel = wp.vec2(-wp.abs(vel[0]) * restitution, vel[1])
        
        if pos[1] < origin_y:
            pos = wp.vec2(pos[0], origin_y)
            vel = wp.vec2(vel[0], wp.abs(vel[1]) * restitution)
        elif pos[1] > origin_y + world_h:
            pos = wp.vec2(pos[0], origin_y + world_h)
            vel = wp.vec2(vel[0], -wp.abs(vel[1]) * restitution)
        
        x[tid] = pos
        v[tid] = vel
        return
    
    # Sample SDF at particle position
    sdf_value = bilinear_sample(sdf, px, py, width, height) / resolution
    
    # Check collision (negative SDF = inside wall)
    if sdf_value < 0.0:
        # Get surface normal from gradient
        grad_x = bilinear_sample(sdf_grad_x, px, py, width, height)
        grad_y = bilinear_sample(sdf_grad_y, px, py, width, height)
        
        # Normalize gradient to get normal (pointing outward from wall)
        grad_len = wp.sqrt(grad_x * grad_x + grad_y * grad_y)
        if grad_len > 1e-6:
            normal = wp.vec2(grad_x / grad_len, grad_y / grad_len)
        else:
            normal = wp.vec2(0.0, 1.0)  # Default up
        
        # Push particle out of wall along normal
        penetration = -sdf_value
        pos = pos + normal * (penetration + 0.01)
        
        # Reflect velocity component into wall
        v_normal = wp.dot(vel, normal)
        if v_normal < 0.0:
            vel = vel - normal * v_normal * (1.0 + restitution)
        
        x[tid] = pos
        v[tid] = vel


@wp.kernel
def apply_sdf_boundary_with_friction_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    sdf: wp.array2d(dtype=float),
    sdf_grad_x: wp.array2d(dtype=float),
    sdf_grad_y: wp.array2d(dtype=float),
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    restitution: float,
    forward_dir: wp.vec2,
    ratchet_enabled: int,
):
    """
    Apply SDF boundary conditions with optional ratchet friction.
    
    Ratchet mechanism (when enabled):
        - Moving in forward direction: slides freely (no friction)
        - Moving backward: stops tangential motion (ratchet grip)
    
    This enables locomotion in a preferred direction.
    
    Args:
        x: Particle positions [N, vec2]
        v: Particle velocities [N, vec2]
        sdf: Signed distance field
        sdf_grad_x, sdf_grad_y: SDF gradient components
        resolution: Pixels per world unit
        origin_x, origin_y: World coordinates of SDF origin
        width, height: SDF dimensions in pixels
        restitution: Bounce coefficient
        forward_dir: Preferred locomotion direction (normalized)
        ratchet_enabled: 1 = enable ratchet friction, 0 = no friction
    """
    tid = wp.tid()
    
    pos = x[tid]
    vel = v[tid]
    
    # Convert world position to pixel coordinates
    px = (pos[0] - origin_x) * resolution
    py = (pos[1] - origin_y) * resolution
    
    # Check if within SDF bounds
    if px < 0.0 or px >= float(width) or py < 0.0 or py >= float(height):
        # Outside bounds - clamp and dampen
        world_w = float(width) / resolution
        world_h = float(height) / resolution
        
        if pos[0] < origin_x:
            pos = wp.vec2(origin_x, pos[1])
            vel = wp.vec2(wp.abs(vel[0]) * 0.5, vel[1])
        elif pos[0] > origin_x + world_w:
            pos = wp.vec2(origin_x + world_w, pos[1])
            vel = wp.vec2(-wp.abs(vel[0]) * 0.5, vel[1])
        
        if pos[1] < origin_y:
            pos = wp.vec2(pos[0], origin_y)
            vel = wp.vec2(vel[0], wp.abs(vel[1]) * 0.5)
        elif pos[1] > origin_y + world_h:
            pos = wp.vec2(pos[0], origin_y + world_h)
            vel = wp.vec2(vel[0], -wp.abs(vel[1]) * 0.5)
        
        x[tid] = pos
        v[tid] = vel
        return
    
    # Sample SDF at particle position
    sdf_value = bilinear_sample(sdf, px, py, width, height) / resolution
    
    # Check collision (negative SDF = inside wall)
    if sdf_value < 0.0:
        # Get surface normal from gradient
        grad_x = bilinear_sample(sdf_grad_x, px, py, width, height)
        grad_y = bilinear_sample(sdf_grad_y, px, py, width, height)
        
        # Normalize gradient to get normal
        grad_len = wp.sqrt(grad_x * grad_x + grad_y * grad_y)
        if grad_len > 1e-6:
            normal = wp.vec2(grad_x / grad_len, grad_y / grad_len)
        else:
            normal = wp.vec2(0.0, 1.0)
        
        # Push particle out of wall
        penetration = -sdf_value
        pos = pos + normal * (penetration + 0.005)
        
        # Decompose velocity into normal and tangential components
        v_normal = wp.dot(vel, normal)
        v_tangent_vec = vel - normal * v_normal
        
        # Reflect normal component (if moving into surface)
        if v_normal < 0.0:
            v_normal = -v_normal * restitution
        
        # Apply ratchet friction on tangent
        if ratchet_enabled == 1:
            # Get tangent direction (perpendicular to normal)
            tangent = wp.vec2(normal[1], -normal[0])
            v_tangent = wp.dot(vel, tangent)
            
            # Check if moving forward or backward
            v_forward = wp.dot(vel, forward_dir)
            
            if v_forward > 0.0:
                # Moving forward: keep tangent velocity (slide)
                pass
            else:
                # Moving backward: stop tangent velocity (ratchet)
                v_tangent = 0.0
            
            vel = normal * v_normal + tangent * v_tangent
        else:
            # No friction - just reflect normal component
            vel = normal * v_normal + v_tangent_vec
        
        x[tid] = pos
        v[tid] = vel



