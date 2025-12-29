#!/usr/bin/env python3
"""
Test and visualize the WorldMap collision system.

This script:
1. Creates test maps of various types
2. Visualizes the bitmap and SDF
3. Simulates particles bouncing around to verify collision detection
4. Tests the ratchet friction mechanism
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_map import WorldMap


def test_basic_maps():
    """Create and visualize all test map types."""
    print("=" * 60)
    print("Testing basic map creation and visualization")
    print("=" * 60)
    
    map_types = ["tunnel", "chamber", "hills", "maze"]
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    for map_type in map_types:
        print(f"\nCreating {map_type} map...")
        world_map = WorldMap.create_test_map(map_type=map_type)
        
        print(f"  Bitmap shape: {world_map.bitmap.shape}")
        print(f"  World size: {world_map.world_size}")
        print(f"  Resolution: {world_map.resolution} px/unit")
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"test_{map_type}.png")
        world_map.save_visualization(viz_path, show_sdf=True)


def test_collision_detection():
    """Test collision detection and normal computation."""
    print("\n" + "=" * 60)
    print("Testing collision detection")
    print("=" * 60)
    
    # Create a simple tunnel map
    world_map = WorldMap.create_test_map(map_type="tunnel")
    
    # Test points
    test_points = [
        np.array([5.0, 5.0]),   # Should be passable (middle of tunnel)
        np.array([5.0, 0.5]),   # Should be in wall (near floor)
        np.array([5.0, 9.5]),   # Should be in wall (near ceiling)
        np.array([10.0, 5.0]),  # Should be passable
    ]
    
    for pos in test_points:
        passable = world_map.is_passable(pos)
        sdf = world_map.get_sdf_value(pos)
        normal = world_map.get_normal(pos)
        
        status = "PASSABLE" if passable else "WALL"
        print(f"  Position {pos}: {status}, SDF={sdf:.3f}, Normal={normal}")


def test_particle_simulation():
    """
    Simulate particles bouncing in the world to verify collision handling.
    Creates an animated visualization.
    """
    print("\n" + "=" * 60)
    print("Testing particle simulation with collisions")
    print("=" * 60)
    
    # Create world map
    world_map = WorldMap.create_test_map(map_type="tunnel")
    
    # Initialize particles
    n_particles = 20
    np.random.seed(42)
    
    # Start particles in the middle of the tunnel
    positions = np.zeros((n_particles, 2))
    positions[:, 0] = np.random.uniform(2, 18, n_particles)  # X spread across tunnel
    positions[:, 1] = np.random.uniform(3, 7, n_particles)   # Y in passable area
    
    # Random initial velocities
    velocities = np.random.randn(n_particles, 2) * 0.5
    velocities[:, 0] += 0.3  # Bias toward right for locomotion
    
    # Forward direction for ratchet
    forward_direction = np.array([1.0, 0.0])
    
    # Simulation parameters
    dt = 0.05
    gravity = np.array([0.0, -2.0])
    n_steps = 200
    
    # Store history for animation
    history = [positions.copy()]
    
    print("  Running simulation...")
    for step in range(n_steps):
        # Apply gravity
        velocities += gravity * dt
        
        # Update positions
        positions += velocities * dt
        
        # Resolve collisions
        for i in range(n_particles):
            new_pos, new_vel = world_map.resolve_collision(
                positions[i],
                velocities[i],
                forward_direction=forward_direction,
                restitution=0.3,
                ratchet_friction=True
            )
            positions[i] = new_pos
            velocities[i] = new_vel
        
        history.append(positions.copy())
    
    print(f"  Simulation complete: {n_steps} steps")
    
    # Create animation
    print("  Creating animation...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show world map
    ax.imshow(np.flipud(world_map.bitmap), cmap='gray', origin='lower',
              extent=[0, world_map.world_size[0], 0, world_map.world_size[1]],
              alpha=0.8)
    
    # Initialize particle scatter
    scatter = ax.scatter([], [], c='red', s=50, zorder=5)
    
    ax.set_xlim(0, world_map.world_size[0])
    ax.set_ylim(0, world_map.world_size[1])
    ax.set_title('Particle Simulation with Ratchet Friction')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    # Add arrow showing forward direction
    ax.annotate('Forward', xy=(15, 8), xytext=(12, 8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, color='green')
    
    def update(frame):
        positions = history[frame]
        scatter.set_offsets(positions)
        ax.set_title(f'Particle Simulation - Step {frame}/{n_steps}')
        return scatter,
    
    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=50, blit=True)
    
    # Save animation
    output_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(output_dir, "test_particle_simulation.gif")
    
    try:
        anim.save(gif_path, writer='pillow', fps=20)
        print(f"  Saved animation to: {gif_path}")
    except Exception as e:
        print(f"  Could not save animation: {e}")
        # Save final frame as static image instead
        png_path = os.path.join(output_dir, "test_particle_simulation.png")
        update(len(history) - 1)
        plt.savefig(png_path, dpi=150)
        print(f"  Saved final frame to: {png_path}")
    
    plt.close()


def test_ratchet_friction():
    """
    Specifically test the ratchet friction mechanism.
    Particles moving forward should slide, backward should stop.
    """
    print("\n" + "=" * 60)
    print("Testing ratchet friction mechanism")
    print("=" * 60)
    
    # Create simple flat ground
    bitmap = np.ones((50, 200), dtype=np.float32)
    bitmap[:10, :] = 0  # Ground
    world_map = WorldMap(bitmap=bitmap, resolution=10.0)
    
    forward_direction = np.array([1.0, 0.0])
    
    # Test cases
    test_cases = [
        {"name": "Moving forward (right)", "vel": np.array([1.0, -0.5])},
        {"name": "Moving backward (left)", "vel": np.array([-1.0, -0.5])},
        {"name": "Moving straight down", "vel": np.array([0.0, -1.0])},
    ]
    
    for case in test_cases:
        pos = np.array([10.0, 0.8])  # Just below ground
        vel = case["vel"].copy()
        
        new_pos, new_vel = world_map.resolve_collision(
            pos, vel,
            forward_direction=forward_direction,
            restitution=0.0,
            ratchet_friction=True
        )
        
        print(f"\n  {case['name']}:")
        print(f"    Initial: pos={pos}, vel={vel}")
        print(f"    After:   pos={new_pos}, vel={new_vel}")


def test_normal_visualization():
    """Visualize surface normals across the map."""
    print("\n" + "=" * 60)
    print("Testing normal field visualization")
    print("=" * 60)
    
    world_map = WorldMap.create_test_map(map_type="hills")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Show bitmap
    ax.imshow(np.flipud(world_map.bitmap), cmap='gray', origin='lower',
              extent=[0, world_map.world_size[0], 0, world_map.world_size[1]],
              alpha=0.7)
    
    # Sample normals near the surface
    x_samples = np.linspace(0.5, world_map.world_size[0] - 0.5, 40)
    
    for x in x_samples:
        # Find surface by scanning y
        for y in np.linspace(0.1, world_map.world_size[1] - 0.1, 50):
            pos = np.array([x, y])
            sdf = world_map.get_sdf_value(pos)
            
            if abs(sdf) < 0.3:  # Near surface
                normal = world_map.get_normal(pos)
                ax.arrow(x, y, normal[0] * 0.3, normal[1] * 0.3,
                        head_width=0.1, head_length=0.05, fc='red', ec='red')
                break
    
    ax.set_xlim(0, world_map.world_size[0])
    ax.set_ylim(0, world_map.world_size[1])
    ax.set_title('Surface Normals (Hills Map)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, "test_normals.png"), dpi=150)
    plt.close()
    print("  Saved normal visualization to: test_normals.png")


def test_custom_image():
    """Test loading a custom bitmap image if one exists."""
    print("\n" + "=" * 60)
    print("Testing custom image loading")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    custom_path = os.path.join(output_dir, "custom_map.png")
    
    if os.path.exists(custom_path):
        print(f"  Loading custom map from: {custom_path}")
        world_map = WorldMap(image_path=custom_path, resolution=5.0)
        world_map.save_visualization(
            os.path.join(output_dir, "test_custom.png"),
            show_sdf=True
        )
    else:
        print(f"  No custom map found at: {custom_path}")
        print("  To test custom image loading, create a PNG file at that path.")
        print("  (White = passable, Black = walls)")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WORLD MAP COLLISION SYSTEM TESTS")
    print("=" * 60)
    
    test_basic_maps()
    test_collision_detection()
    test_ratchet_friction()
    test_normal_visualization()
    test_particle_simulation()
    test_custom_image()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nCheck the world_map folder for generated images:")
    print("  - test_tunnel.png, test_chamber.png, etc. (map visualizations)")
    print("  - test_normals.png (surface normal visualization)")
    print("  - test_particle_simulation.gif (animated collision test)")


if __name__ == "__main__":
    main()

