"""
WorldMap: Bitmap-based collision detection for arbitrary 2D terrain.

Uses a bitmap image where:
    - White (255) = passable area (soft body can move)
    - Black (0) = walls/obstacles

Computes a Signed Distance Field (SDF) for:
    - Efficient collision detection
    - Surface normal computation at boundaries
    - Proper reflection and friction handling
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from typing import Tuple, Optional
import os


class WorldMap:
    """
    Bitmap-based world map for 2D collision detection.
    
    Attributes:
        bitmap: Binary array (1 = passable, 0 = wall)
        sdf: Signed distance field (positive inside passable, negative in walls)
        resolution: Pixels per world unit
        world_size: (width, height) in world coordinates
    """
    
    def __init__(
        self,
        image_path: Optional[str] = None,
        bitmap: Optional[np.ndarray] = None,
        resolution: float = 10.0,  # pixels per world unit
        origin: Tuple[float, float] = (0.0, 0.0),
        threshold: int = 128,  # pixel value threshold for wall/passable
    ):
        """
        Initialize WorldMap from image file or numpy array.
        
        Args:
            image_path: Path to bitmap image (PNG, JPG, etc.)
            bitmap: Alternatively, provide a numpy array directly
            resolution: Pixels per world unit (higher = more detail)
            origin: World coordinates of bottom-left corner
            threshold: Pixel value threshold (< threshold = wall)
        """
        self.resolution = resolution
        self.origin = np.array(origin)
        
        if image_path is not None:
            self._load_from_image(image_path, threshold)
        elif bitmap is not None:
            self._load_from_array(bitmap)
        else:
            raise ValueError("Must provide either image_path or bitmap array")
        
        # Compute SDF for collision detection and normal computation
        self._compute_sdf()
        
        # Compute world size
        self.world_size = (
            self.bitmap.shape[1] / self.resolution,
            self.bitmap.shape[0] / self.resolution
        )
        
    def _load_from_image(self, image_path: str, threshold: int):
        """Load bitmap from image file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        
        # Flip Y axis so origin is bottom-left (standard physics convention)
        img_array = np.flipud(img_array)
        
        # Convert to binary: 1 = passable (white), 0 = wall (black)
        self.bitmap = (img_array >= threshold).astype(np.float32)
        
    def _load_from_array(self, bitmap: np.ndarray):
        """Load from numpy array."""
        if bitmap.ndim != 2:
            raise ValueError("Bitmap must be 2D array")
        # Ensure binary
        self.bitmap = (bitmap > 0.5).astype(np.float32)
    
    def _compute_sdf(self):
        """
        Compute Signed Distance Field from bitmap.
        
        SDF > 0: inside passable area (distance to nearest wall)
        SDF < 0: inside wall (distance to nearest passable)
        SDF = 0: exactly on boundary
        """
        # Distance transform for passable regions
        dist_passable = ndimage.distance_transform_edt(self.bitmap)
        
        # Distance transform for wall regions
        dist_wall = ndimage.distance_transform_edt(1 - self.bitmap)
        
        # Signed distance: positive in passable, negative in walls
        self.sdf = (dist_passable - dist_wall).astype(np.float32)
        
        # Also compute gradient for normals (pointing toward passable area)
        # Using Sobel for smoother gradients
        self.sdf_grad_x = ndimage.sobel(self.sdf, axis=1).astype(np.float32)
        self.sdf_grad_y = ndimage.sobel(self.sdf, axis=0).astype(np.float32)
        
    def world_to_pixel(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to pixel indices."""
        local_pos = world_pos - self.origin
        px = int(local_pos[0] * self.resolution)
        py = int(local_pos[1] * self.resolution)
        return px, py
    
    def world_to_pixel_float(self, world_pos: np.ndarray) -> Tuple[float, float]:
        """Convert world coordinates to floating point pixel coordinates."""
        local_pos = world_pos - self.origin
        px = local_pos[0] * self.resolution
        py = local_pos[1] * self.resolution
        return px, py
    
    def pixel_to_world(self, px: int, py: int) -> np.ndarray:
        """Convert pixel indices to world coordinates."""
        world_x = px / self.resolution + self.origin[0]
        world_y = py / self.resolution + self.origin[1]
        return np.array([world_x, world_y])
    
    def is_inside_bounds(self, world_pos: np.ndarray) -> bool:
        """Check if position is within map bounds."""
        px, py = self.world_to_pixel(world_pos)
        return (0 <= px < self.bitmap.shape[1] and 
                0 <= py < self.bitmap.shape[0])
    
    def is_passable(self, world_pos: np.ndarray) -> bool:
        """Check if position is in passable area (not a wall)."""
        if not self.is_inside_bounds(world_pos):
            return False
        px, py = self.world_to_pixel(world_pos)
        return self.bitmap[py, px] > 0.5
    
    def get_sdf_value(self, world_pos: np.ndarray) -> float:
        """
        Get SDF value at world position (with bilinear interpolation).
        
        Returns:
            Positive = inside passable area
            Negative = inside wall
        """
        px, py = self.world_to_pixel_float(world_pos)
        
        # Clamp to valid range
        px = np.clip(px, 0, self.sdf.shape[1] - 1.001)
        py = np.clip(py, 0, self.sdf.shape[0] - 1.001)
        
        # Bilinear interpolation
        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, self.sdf.shape[1] - 1), min(y0 + 1, self.sdf.shape[0] - 1)
        
        fx, fy = px - x0, py - y0
        
        v00 = self.sdf[y0, x0]
        v01 = self.sdf[y0, x1]
        v10 = self.sdf[y1, x0]
        v11 = self.sdf[y1, x1]
        
        # Interpolate
        v0 = v00 * (1 - fx) + v01 * fx
        v1 = v10 * (1 - fx) + v11 * fx
        
        return (v0 * (1 - fy) + v1 * fy) / self.resolution  # Convert to world units
    
    def get_normal(self, world_pos: np.ndarray) -> np.ndarray:
        """
        Get surface normal at world position (pointing toward passable area).
        
        Uses gradient of SDF for smooth normals even at edges.
        """
        px, py = self.world_to_pixel_float(world_pos)
        
        # Clamp to valid range
        px = np.clip(px, 0, self.sdf_grad_x.shape[1] - 1.001)
        py = np.clip(py, 0, self.sdf_grad_x.shape[0] - 1.001)
        
        # Bilinear interpolation for gradient components
        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, self.sdf_grad_x.shape[1] - 1), min(y0 + 1, self.sdf_grad_x.shape[0] - 1)
        
        fx, fy = px - x0, py - y0
        
        # Interpolate gradient X
        gx00, gx01 = self.sdf_grad_x[y0, x0], self.sdf_grad_x[y0, x1]
        gx10, gx11 = self.sdf_grad_x[y1, x0], self.sdf_grad_x[y1, x1]
        gx0 = gx00 * (1 - fx) + gx01 * fx
        gx1 = gx10 * (1 - fx) + gx11 * fx
        grad_x = gx0 * (1 - fy) + gx1 * fy
        
        # Interpolate gradient Y
        gy00, gy01 = self.sdf_grad_y[y0, x0], self.sdf_grad_y[y0, x1]
        gy10, gy11 = self.sdf_grad_y[y1, x0], self.sdf_grad_y[y1, x1]
        gy0 = gy00 * (1 - fx) + gy01 * fx
        gy1 = gy10 * (1 - fx) + gy11 * fx
        grad_y = gy0 * (1 - fy) + gy1 * fy
        
        # Normalize
        length = np.sqrt(grad_x**2 + grad_y**2)
        if length < 1e-6:
            return np.array([0.0, 1.0])  # Default up if no gradient
        
        return np.array([grad_x / length, grad_y / length])
    
    def get_tangent(self, world_pos: np.ndarray, forward_sign: float = 1.0) -> np.ndarray:
        """
        Get surface tangent at world position.
        
        Args:
            forward_sign: +1.0 for "right" direction, -1.0 for "left"
        """
        normal = self.get_normal(world_pos)
        # Tangent is perpendicular to normal: rotate 90 degrees
        tangent = np.array([normal[1], -normal[0]]) * forward_sign
        return tangent
    
    def resolve_collision(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        forward_direction: np.ndarray = np.array([1.0, 0.0]),
        restitution: float = 0.0,
        ratchet_friction: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve collision with walls, applying reflection and friction.
        
        Args:
            pos: Current position
            vel: Current velocity
            forward_direction: Locomotion direction for ratchet friction
            restitution: Bounce coefficient (0 = no bounce, 1 = full bounce)
            ratchet_friction: If True, apply ratchet mechanism
            
        Returns:
            (new_pos, new_vel): Corrected position and velocity
        """
        sdf_value = self.get_sdf_value(pos)
        
        # No collision if inside passable area
        if sdf_value >= 0:
            return pos, vel
        
        # Get surface normal at this point
        normal = self.get_normal(pos)
        
        # Push position out of wall along normal
        penetration = -sdf_value
        new_pos = pos + normal * (penetration + 0.01 / self.resolution)
        
        # Decompose velocity into normal and tangential components
        v_normal = np.dot(vel, normal)
        v_tangent_vec = vel - v_normal * normal
        
        # Reflect normal component
        if v_normal < 0:
            v_normal = -v_normal * restitution
        
        # Tangent direction (for ratchet)
        tangent = np.array([normal[1], -normal[0]])
        v_tangent = np.dot(vel, tangent)
        
        if ratchet_friction:
            # Project velocity onto forward direction to check if moving forward
            v_forward = np.dot(vel, forward_direction)
            
            if v_forward > 0:
                # Moving forward: no friction, keep tangent velocity
                pass
            else:
                # Moving backward: stop tangential motion (ratchet)
                v_tangent = 0.0
        
        # Reconstruct velocity
        new_vel = normal * v_normal + tangent * v_tangent
        
        return new_pos, new_vel
    
    def to_numpy_arrays(self) -> dict:
        """
        Export data as numpy arrays for use with Warp kernels.
        
        Returns dict with:
            - bitmap: Binary collision mask
            - sdf: Signed distance field
            - sdf_grad_x, sdf_grad_y: Gradient components
            - resolution, origin, world_size: Metadata
        """
        return {
            'bitmap': self.bitmap.copy(),
            'sdf': self.sdf.copy(),
            'sdf_grad_x': self.sdf_grad_x.copy(),
            'sdf_grad_y': self.sdf_grad_y.copy(),
            'resolution': self.resolution,
            'origin': self.origin.copy(),
            'world_size': self.world_size,
        }
    
    @staticmethod
    def create_test_map(
        width: int = 200,
        height: int = 100,
        map_type: str = "tunnel"
    ) -> 'WorldMap':
        """
        Create a test map for development/debugging.
        
        Args:
            width, height: Map dimensions in pixels
            map_type: One of "tunnel", "chamber", "hills", "maze"
        """
        bitmap = np.ones((height, width), dtype=np.float32)
        
        if map_type == "tunnel":
            # Simple tunnel with ceiling and floor
            bitmap[:15, :] = 0  # Floor
            bitmap[-15:, :] = 0  # Ceiling
            # Some obstacles
            bitmap[15:40, 80:100] = 0
            bitmap[60:85, 140:160] = 0
            
        elif map_type == "chamber":
            # Multiple connected chambers
            bitmap[:10, :] = 0  # Floor
            bitmap[-10:, :] = 0  # Ceiling
            bitmap[:, :10] = 0  # Left wall
            bitmap[:, -10:] = 0  # Right wall
            # Internal walls with openings
            bitmap[10:70, 60:70] = 0
            bitmap[30:50, 60:70] = 1  # Opening
            bitmap[30:90, 130:140] = 0
            bitmap[50:70, 130:140] = 1  # Opening
            
        elif map_type == "hills":
            # Hilly terrain (ground only)
            for x in range(width):
                hill_height = int(20 + 15 * np.sin(x * 0.05) + 10 * np.sin(x * 0.02))
                bitmap[:hill_height, x] = 0
                
        elif map_type == "maze":
            # Simple maze pattern
            bitmap[:5, :] = 0  # Floor
            bitmap[-5:, :] = 0  # Ceiling
            bitmap[:, :5] = 0  # Left wall
            bitmap[:, -5:] = 0  # Right wall
            # Maze walls
            for i in range(1, 6):
                x_start = i * 30
                if i % 2 == 1:
                    bitmap[5:70, x_start:x_start+5] = 0
                else:
                    bitmap[30:95, x_start:x_start+5] = 0
        else:
            raise ValueError(f"Unknown map type: {map_type}")
        
        return WorldMap(bitmap=bitmap, resolution=10.0)
    
    def save_visualization(self, filepath: str, show_sdf: bool = False):
        """Save map visualization to image file."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2 if show_sdf else 1, figsize=(12 if show_sdf else 6, 6))
        
        if show_sdf:
            ax1, ax2 = axes
        else:
            ax1 = axes
        
        # Show bitmap
        ax1.imshow(np.flipud(self.bitmap), cmap='gray', origin='lower',
                   extent=[self.origin[0], self.origin[0] + self.world_size[0],
                           self.origin[1], self.origin[1] + self.world_size[1]])
        ax1.set_title('World Map (White=Passable, Black=Wall)')
        ax1.set_xlabel('X (world units)')
        ax1.set_ylabel('Y (world units)')
        
        if show_sdf:
            # Show SDF
            sdf_display = np.flipud(self.sdf)
            im = ax2.imshow(sdf_display, cmap='RdBu', origin='lower',
                           extent=[self.origin[0], self.origin[0] + self.world_size[0],
                                   self.origin[1], self.origin[1] + self.world_size[1]])
            ax2.set_title('Signed Distance Field')
            ax2.set_xlabel('X (world units)')
            ax2.set_ylabel('Y (world units)')
            plt.colorbar(im, ax=ax2, label='Distance (pixels)')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved visualization to: {filepath}")

