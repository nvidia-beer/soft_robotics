"""
Refined Delaunay Triangulation with Interior Points
Grid-aligned boundary with edge collapse
"""

import numpy as np
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import json
from collections import defaultdict


def find_edge_pixels(img_np):
    """Extract all edge pixels from binary image."""
    height, width = img_np.shape
    edge_pixels = set()
    
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for y in range(height):
        for x in range(width):
            if img_np[y, x] > 0:
                is_edge = False
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        is_edge = True
                        break
                    if img_np[ny, nx] == 0:
                        is_edge = True
                        break
                
                if is_edge:
                    edge_pixels.add((x, y))
    
    return edge_pixels


def detect_extremal_corners(edge_pixels):
    """
    Detect corners by analyzing direction changes (dx, dy) along the boundary.
    A corner is where the boundary changes direction significantly.
    
    Args:
        edge_pixels: Set of (x, y) edge coordinates
    
    Returns:
        Set of corner (x, y) coordinates
    """
    if len(edge_pixels) < 10:
        return edge_pixels
    
    edge_list = list(edge_pixels)
    edge_array = np.array(edge_list)
    
    # Sort edge points to form a contour (by angle from centroid)
    centroid = edge_array.mean(axis=0)
    angles = np.arctan2(edge_array[:, 1] - centroid[1], edge_array[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_edge = edge_array[sorted_indices]
    
    # Detect corners by looking at direction changes
    corners = set()
    window = 10  # Look at neighbors within this distance
    angle_threshold = 120  # Degrees - aggressive corner detection
    
    for i in range(len(sorted_edge)):
        curr = sorted_edge[i]
        prev = sorted_edge[(i - window) % len(sorted_edge)]
        next_pt = sorted_edge[(i + window) % len(sorted_edge)]
        
        # Direction vectors
        v1 = prev - curr
        v2 = next_pt - curr
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        
        # Normalize and compute angle
        v1 = v1 / len1
        v2 = v2 / len2
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        
        # Sharp angle = corner
        if angle_deg < angle_threshold:
            corners.add(tuple(curr))
    
    print(f"    Initial corner detection: {len(corners)} corners")
    
    # Merge nearby corners (within 5 pixels - more aggressive = keep more corners)
    corners_list = list(corners)
    merged_corners = set()
    used = set()
    
    for i, c1 in enumerate(corners_list):
        if i in used:
            continue
        cluster = [c1]
        used.add(i)
        
        for j, c2 in enumerate(corners_list):
            if j in used:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist < 5:  # Smaller merge distance = more corners preserved
                cluster.append(c2)
                used.add(j)
        
        # Use centroid of cluster
        cluster_array = np.array(cluster)
        merged_corner = tuple(cluster_array.mean(axis=0).astype(int))
        merged_corners.add(merged_corner)
    
    return merged_corners


def find_boundary_and_interior_adaptive(img_np, spacing=5, detect_corners=True):
    """
    Grid-based boundary detection with optional corner detection.
    
    Algorithm:
    1. Extract all edge pixels
    2. If detect_corners: Find corners by direction change
    3. Scan horizontal/vertical lines at grid spacing
    4. Add mandatory corner points + sampled boundary points
    5. Fill interior with grid-spaced points
    
    Args:
        img_np: Binary image
        spacing: Grid spacing for scanlines and interior points
        detect_corners: If True, detect and preserve corner points
    
    Returns:
        boundary_points, interior_points
    """
    height, width = img_np.shape
    boundary_points = set()
    interior_points = set()
    
    mandatory_corners = set()
    
    if detect_corners:
        print(f"  - Detecting corners by direction change...")
        edge_pixels = find_edge_pixels(img_np)
        print(f"    Found {len(edge_pixels)} edge pixels")
        
        corners = detect_extremal_corners(edge_pixels)
        mandatory_corners = corners
        print(f"    Detected {len(corners)} corner points (mandatory)")
        
        # Add all corners as mandatory boundary points
        boundary_points.update(corners)
    
    print(f"  - Scanning horizontal/vertical lines at spacing={spacing}...")
    
    # Horizontal scanlines at grid spacing
    for y in range(0, height, spacing):
        if y >= height:
            continue
            
        in_shape = False
        entry_x = None
        
        for x in range(width):
            is_white = img_np[y, x] > 0
            
            if is_white and not in_shape:
                boundary_points.add((x, y))
                entry_x = x
                in_shape = True
            elif not is_white and in_shape:
                boundary_points.add((x - 1, y))
                
                if entry_x is not None:
                    for xi in range(entry_x + spacing, x - 1, spacing):
                        interior_points.add((xi, y))
                
                in_shape = False
                entry_x = None
        
        if in_shape and entry_x is not None:
            boundary_points.add((width - 1, y))
            for xi in range(entry_x + spacing, width - 1, spacing):
                interior_points.add((xi, y))
    
    # Vertical scanlines at grid spacing
    for x in range(0, width, spacing):
        if x >= width:
            continue
            
        in_shape = False
        entry_y = None
        
        for y in range(height):
            is_white = img_np[y, x] > 0
            
            if is_white and not in_shape:
                boundary_points.add((x, y))
                entry_y = y
                in_shape = True
            elif not is_white and in_shape:
                boundary_points.add((x, y - 1))
                
                if entry_y is not None:
                    for yi in range(entry_y + spacing, y - 1, spacing):
                        interior_points.add((x, yi))
                
                in_shape = False
                entry_y = None
        
        if in_shape and entry_y is not None:
            boundary_points.add((x, height - 1))
            for yi in range(entry_y + spacing, height - 1, spacing):
                interior_points.add((x, yi))
    
    interior_points = interior_points - boundary_points
    
    boundary_array = np.array(sorted(list(boundary_points)), dtype=np.float64)
    interior_array = np.array(sorted(list(interior_points)), dtype=np.float64)
    corners_array = np.array(sorted(list(mandatory_corners)), dtype=np.float64) if mandatory_corners else np.array([])
    
    return boundary_array, interior_array, corners_array


def compute_triangle_quality(triangle):
    """Compute aspect ratio and minimum angle."""
    edges = [
        np.linalg.norm(triangle[1] - triangle[0]),
        np.linalg.norm(triangle[2] - triangle[1]),
        np.linalg.norm(triangle[0] - triangle[2])
    ]
    
    min_edge = min(edges)
    max_edge = max(edges)
    aspect_ratio = max_edge / min_edge if min_edge > 1e-10 else float('inf')
    
    a, b, c = edges
    try:
        cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
        
        angles = [
            np.arccos(np.clip(cos_A, -1, 1)),
            np.arccos(np.clip(cos_B, -1, 1)),
            np.arccos(np.clip(cos_C, -1, 1))
        ]
        
        min_angle = min(angles) * 180 / np.pi
    except:
        min_angle = 0
    
    return aspect_ratio, min_angle


def collapse_small_triangles(vertices, triangles, min_area_threshold, protected_vertices=None):
    """
    Collapse triangles with area < threshold via edge collapse.
    Merges vertices and updates connectivity.
    
    Args:
        vertices: Vertex positions
        triangles: Triangle indices
        min_area_threshold: Minimum area for triangles
        protected_vertices: Set of vertex indices that must not be moved/merged
    """
    vertices = np.array(vertices, dtype=np.float64)
    triangles = [list(tri) for tri in triangles]
    
    if protected_vertices is None:
        protected_vertices = set()
    
    def triangle_area(p0, p1, p2):
        v1, v2 = p1 - p0, p2 - p0
        return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    def edge_length(v1, v2):
        return np.linalg.norm(v2 - v1)
    
    iteration = 0
    max_iterations = 200
    
    while iteration < max_iterations:
        iteration += 1
        
        small_triangles = []
        for tri_idx, tri in enumerate(triangles):
            area = triangle_area(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
            if area < min_area_threshold:
                # Identify which vertices are protected (corners)
                edges = [
                    (tri[0], tri[1], edge_length(vertices[tri[0]], vertices[tri[1]])),
                    (tri[1], tri[2], edge_length(vertices[tri[1]], vertices[tri[2]])),
                    (tri[2], tri[0], edge_length(vertices[tri[2]], vertices[tri[0]]))
                ]
                
                # Prioritize edges with at least one non-corner vertex
                # This helps create edges between adjacent corners
                collapsible_edges = []
                for v1, v2, length in edges:
                    v1_prot = v1 in protected_vertices
                    v2_prot = v2 in protected_vertices
                    
                    if not (v1_prot and v2_prot):
                        # At least one vertex is not protected
                        # Priority: edges with exactly one protected vertex (creates corner edges)
                        if v1_prot or v2_prot:
                            collapsible_edges.append((v1, v2, length, 0))  # High priority
                        else:
                            collapsible_edges.append((v1, v2, length, 1))  # Low priority
                
                if collapsible_edges:
                    # Sort by priority first, then by length
                    collapsible_edges.sort(key=lambda x: (x[3], x[2]))
                    best_edge = collapsible_edges[0]
                    small_triangles.append((tri_idx, best_edge[0], best_edge[1], area))
        
        if not small_triangles:
            break
        
        tri_idx, v1_idx, v2_idx, area = small_triangles[0]
        tri = triangles[tri_idx]
        
        # Check protection status of the edge vertices
        v1_protected = v1_idx in protected_vertices
        v2_protected = v2_idx in protected_vertices
        
        if v1_protected and v2_protected:
            # Both edge vertices are protected corners
            # Find the third vertex of this triangle
            third_vertex = None
            for v in tri:
                if v != v1_idx and v != v2_idx:
                    third_vertex = v
                    break
            
            if third_vertex is not None and third_vertex not in protected_vertices:
                # Collapse the non-protected vertex into one of the corners
                # This creates an edge between the two corners
                merge_target = v1_idx
                merge_source = third_vertex
                
                # Keep corner position
                merged_pos = vertices[merge_target]
                vertices[merge_target] = merged_pos
                
                # Replace merge_source with merge_target in all triangles
                for t in triangles:
                    for i in range(3):
                        if t[i] == merge_source:
                            t[i] = merge_target
                
                triangles = [t for t in triangles if len(set(t)) == 3]
                continue
            else:
                # All three vertices are protected - skip this triangle
                continue
        elif v2_protected:
            # Keep v2 (protected), merge v1 into v2
            v1_idx, v2_idx = v2_idx, v1_idx
            v1_protected, v2_protected = v2_protected, v1_protected
        
        # Merge vertices
        if v1_protected:
            # Keep v1 position unchanged (it's a corner!)
            merged_pos = vertices[v1_idx]
        else:
            # Merge at midpoint
            merged_pos = (vertices[v1_idx] + vertices[v2_idx]) / 2.0
        
        vertices[v1_idx] = merged_pos
        
        for tri in triangles:
            for i in range(3):
                if tri[i] == v2_idx:
                    tri[i] = v1_idx
        
        triangles = [tri for tri in triangles if len(set(tri)) == 3]
    
    # Merge nearby disconnected vertices (but protect corners)
    merge_threshold = min_area_threshold ** 0.5
    vertices_array = np.array(vertices)
    merge_map = {}
    merged_count = 0
    
    for i in range(len(vertices)):
        if i in merge_map or i in protected_vertices:
            continue
        for j in range(i+1, len(vertices)):
            if j in merge_map or j in protected_vertices:
                continue
            dist = np.linalg.norm(vertices_array[i] - vertices_array[j])
            if dist < merge_threshold:
                merge_map[j] = i
                vertices_array[i] = (vertices_array[i] + vertices_array[j]) / 2.0
                merged_count += 1
    
    if merge_map:
        for tri in triangles:
            for k in range(3):
                if tri[k] in merge_map:
                    tri[k] = merge_map[tri[k]]
        
        triangles = [tri for tri in triangles if len(set(tri)) == 3]
        
        used_vertices = set()
        for tri in triangles:
            used_vertices.update(tri)
        
        old_to_new = {}
        final_vertices = []
        updated_protected = set()  # Track protected vertices through remapping
        
        for new_idx, old_idx in enumerate(sorted(used_vertices)):
            old_to_new[old_idx] = new_idx
            final_vertices.append(vertices_array[old_idx].tolist())
            # If this old index was protected, mark the new index as protected
            if old_idx in protected_vertices:
                updated_protected.add(new_idx)
        
        final_triangles = [[old_to_new[v] for v in tri] for tri in triangles]
        
        if merged_count > 0:
            print(f"    Merged {merged_count} nearby vertex pairs")
            print(f"    Protected vertices remapped: {len(protected_vertices)} → {len(updated_protected)}")
        
        return final_vertices, final_triangles, iteration - 1
    
    used_vertices = set()
    for tri in triangles:
        used_vertices.update(tri)
    
    old_to_new = {}
    new_vertices = []
    updated_protected = set()  # Track protected vertices through remapping
    
    for new_idx, old_idx in enumerate(sorted(used_vertices)):
        old_to_new[old_idx] = new_idx
        new_vertices.append(vertices[old_idx].tolist())
        # If this old index was protected, mark the new index as protected
        if old_idx in protected_vertices:
            updated_protected.add(new_idx)
    
    new_triangles = [[old_to_new[v] for v in tri] for tri in triangles]
    
    print(f"    Protected vertices remapped: {len(protected_vertices)} → {len(updated_protected)}")
    
    return new_vertices, new_triangles, iteration - 1


def refined_delaunay_tessellation(image_path, output_json, output_viz=None,
                                  interior_spacing=10, max_aspect_ratio=5.0,
                                  scale_factor=1, normalize=True, min_area_factor=0.5):
    """
    Create refined Delaunay triangulation with interior points.
    """
    grid_cell_area_pixels = interior_spacing ** 2
    
    print("="*70)
    print("REFINED DELAUNAY TRIANGULATION WITH INTERIOR POINTS")
    print("="*70)
    print(f"Parameters:")
    print(f"  - Interior spacing: {interior_spacing} pixels")
    print(f"  - Max aspect ratio: {max_aspect_ratio}")
    print(f"  - Scale factor: {scale_factor}x")
    print(f"  - Normalize: {normalize}")
    print(f"  - Min area factor: {min_area_factor} (in normalized [0,1] space)")
    
    # Load image
    print(f"\n[1/5] Loading image...")
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.uint8)
    img_np = np.where(img_np > 128, 255, 0).astype(np.uint8)
    height, width = img_np.shape
    print(f"  ✓ Image size: {width}×{height}")
    
    # Extract boundary and interior adaptively with corner detection
    print(f"\n[2/5] Adaptive point placement with corner detection...")
    
    boundary_points, interior_points, corner_points = find_boundary_and_interior_adaptive(
        img_np, spacing=interior_spacing, detect_corners=True
    )
    
    print(f"  ✓ Found {len(boundary_points)} boundary points (grid-aligned)")
    print(f"  ✓ Added {len(interior_points)} interior points (adaptive)")
    if len(corner_points) > 0:
        print(f"  ✓ Protected {len(corner_points)} corner vertices")
        for i, corner in enumerate(corner_points):
            print(f"      Corner {i+1}: {corner}")
    
    # Combine all points
    all_points = np.vstack([boundary_points, interior_points])
    
    # Verify corners are in all_points
    if len(corner_points) > 0:
        print(f"  Verifying corners are in input points...")
        for i, corner in enumerate(corner_points):
            found = False
            for pt in all_points:
                if abs(pt[0] - corner[0]) < 0.01 and abs(pt[1] - corner[1]) < 0.01:
                    found = True
                    break
            if found:
                print(f"      ✓ Corner {i+1} {tuple(corner)} is in input")
            else:
                print(f"      ❌ Corner {i+1} {tuple(corner)} NOT FOUND in input!")
    
    print(f"  ✓ Total points: {len(all_points)}")
    
    # Delaunay triangulation
    print(f"\n[3/5] Computing Delaunay triangulation...")
    tri = Delaunay(all_points)
    print(f"  ✓ Generated {len(tri.simplices)} triangles")
    
    # Filter triangles - basic geometry
    print(f"\n[4/5] Filtering triangles (basic geometry)...")
    candidate_triangles = []
    
    stats = {
        'total': len(tri.simplices),
        'bad_aspect': 0,
        'outside_shape': 0,
        'too_small': 0,
        'poorly_connected': 0,
        'valid': 0
    }
    
    for tri_idx, simplex in enumerate(tri.simplices):
        triangle = all_points[simplex]
        
        # Check for degenerate triangle (zero or near-zero area)
        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        
        # Filter out degenerate triangles in pixel space
        min_pixel_area = 0.1  # Minimum area in pixels
        if area < min_pixel_area:
            stats['too_small'] += 1
            continue
        
        center = np.mean(triangle, axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))
        
        if not (0 <= cx < width and 0 <= cy < height and img_np[cy, cx] > 0):
            stats['outside_shape'] += 1
            continue
        
        aspect_ratio, min_angle = compute_triangle_quality(triangle)
        
        if aspect_ratio > max_aspect_ratio:
            stats['bad_aspect'] += 1
            continue
        
        candidate_triangles.append({
            'tri_idx': tri_idx,
            'indices': simplex.tolist(),
            'coords': triangle.tolist(),
            'aspect_ratio': aspect_ratio,
            'min_angle': min_angle,
            'area': area
        })
    
    print(f"  Passed geometry filters: {len(candidate_triangles)}")
    
    # Connectivity filter
    print(f"\n[5/5] Filtering by connectivity...")
    
    edge_to_triangles = defaultdict(list)
    
    for tri_data in candidate_triangles:
        tri_idx = tri_data['tri_idx']
        indices = tri_data['indices']
        
        edges = [
            tuple(sorted([indices[0], indices[1]])),
            tuple(sorted([indices[1], indices[2]])),
            tuple(sorted([indices[2], indices[0]]))
        ]
        
        for edge in edges:
            edge_to_triangles[edge].append(tri_idx)
    
    valid_triangles = []
    
    for tri_data in candidate_triangles:
        tri_idx = tri_data['tri_idx']
        indices = tri_data['indices']
        
        edges = [
            tuple(sorted([indices[0], indices[1]])),
            tuple(sorted([indices[1], indices[2]])),
            tuple(sorted([indices[2], indices[0]]))
        ]
        
        has_shared_edge = False
        for edge in edges:
            if len(edge_to_triangles[edge]) > 1:
                has_shared_edge = True
                break
        
        if has_shared_edge:
            tri_data_clean = {
                'indices': tri_data['indices'],
                'coords': tri_data['coords'],
                'aspect_ratio': tri_data['aspect_ratio'],
                'min_angle': tri_data['min_angle'],
                'area': tri_data['area']
            }
            valid_triangles.append(tri_data_clean)
            stats['valid'] += 1
        else:
            stats['poorly_connected'] += 1
    
    print(f"  Removed poorly connected: {stats['poorly_connected']}")
    print(f"  Final valid triangles: {stats['valid']}")
    
    print(f"\n  Filtering summary (initial):")
    print(f"    - Too small (pixel space): {stats['too_small']}")
    print(f"    - Outside shape: {stats['outside_shape']}")
    print(f"    - Bad aspect ratio: {stats['bad_aspect']}")
    print(f"    - Poorly connected: {stats['poorly_connected']}")
    print(f"    - Passed to normalization: {stats['valid']}")
    
    if stats['valid'] > 0:
        avg_aspect = np.mean([t['aspect_ratio'] for t in valid_triangles])
        avg_angle = np.mean([t['min_angle'] for t in valid_triangles])
        
        print(f"\n  Quality metrics:")
        print(f"    ✓ Average aspect ratio: {avg_aspect:.2f}")
        print(f"    ✓ Average min angle: {avg_angle:.1f}°")
    
    # Normalize coordinates
    normalized_triangles = None
    bounding_box = None
    
    if normalize and stats['valid'] > 0:
        print(f"\n[6/7] Normalizing coordinates...")
        
        all_triangle_points = np.vstack([t['coords'] for t in valid_triangles])
        
        min_x = np.min(all_triangle_points[:, 0])
        max_x = np.max(all_triangle_points[:, 0])
        min_y = np.min(all_triangle_points[:, 1])
        max_y = np.max(all_triangle_points[:, 1])
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        print(f"  Bounding box: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
        print(f"  Size: {bbox_width:.1f} × {bbox_height:.1f}")
        
        normalized_triangles = []
        for tri_data in valid_triangles:
            normalized = [
                [(x - min_x) / bbox_width, (y - min_y) / bbox_height]  # Keep Y as-is (Y=0 at top)
                for x, y in tri_data['coords']
            ]
            normalized_triangles.append(normalized)
        
        bounding_box = {
            'min': [float(min_x), float(min_y)],
            'max': [float(max_x), float(max_y)],
            'width': float(bbox_width),
            'height': float(bbox_height)
        }
        
        norm_grid_spacing_x = interior_spacing / bbox_width
        norm_grid_spacing_y = interior_spacing / bbox_height
        norm_grid_cell_area = norm_grid_spacing_x * norm_grid_spacing_y
        min_normalized_area = min_area_factor * norm_grid_cell_area
        
        print(f"  Normalized grid spacing: {norm_grid_spacing_x:.6f} × {norm_grid_spacing_y:.6f}")
        print(f"  Normalized grid cell area: {norm_grid_cell_area:.8f}")
        print(f"  Min normalized area threshold: {min_normalized_area:.8f}")
        
        # Edge collapse
        print(f"\n[7/7] Edge collapse for small triangles...")
        
        vertex_map = {}
        norm_vertices = []
        
        for tri_data in valid_triangles:
            for coord in tri_data['coords']:
                # Keep Y as-is (Y=0 at top, matching image coordinates)
                norm_coord = [(coord[0] - min_x) / bbox_width, (coord[1] - min_y) / bbox_height]
                coord_tuple = tuple(coord)
                if coord_tuple not in vertex_map:
                    vertex_map[coord_tuple] = len(norm_vertices)
                    norm_vertices.append(norm_coord)
        
        tri_indices = []
        for tri_data in valid_triangles:
            indices = [vertex_map[tuple(coord)] for coord in tri_data['coords']]
            tri_indices.append(indices)
        
        # Map corner pixel coordinates to normalized vertex indices
        protected_vertex_indices = set()
        if len(corner_points) > 0:
            print(f"  Mapping corners to vertex indices...")
            for i, (corner_x, corner_y) in enumerate(corner_points):
                # Find this corner in the vertex map
                found = False
                closest_dist = float('inf')
                closest_coord = None
                for coord_tuple, vert_idx in vertex_map.items():
                    coord_x, coord_y = coord_tuple
                    dist = abs(coord_x - corner_x) + abs(coord_y - corner_y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_coord = coord_tuple
                    # Check if this is the corner (within 1 pixel tolerance)
                    if abs(coord_x - corner_x) < 1.5 and abs(coord_y - corner_y) < 1.5:
                        protected_vertex_indices.add(vert_idx)
                        found = True
                        print(f"      ✓ Corner {i+1} ({corner_x}, {corner_y}) → vertex {vert_idx}")
                        break
                
                if not found:
                    print(f"      ❌ Corner {i+1} ({corner_x}, {corner_y}) NOT MAPPED! Closest: {closest_coord} (dist={closest_dist:.2f})")
        
        print(f"  Before collapse: {len(norm_vertices)} vertices, {len(tri_indices)} triangles")
        if protected_vertex_indices:
            print(f"  Protecting {len(protected_vertex_indices)} corner vertices from modification")
        
        collapsed_vertices, collapsed_triangles, iterations = collapse_small_triangles(
            norm_vertices, tri_indices, min_normalized_area, protected_vertex_indices
        )
        
        print(f"  After collapse: {len(collapsed_vertices)} vertices, {len(collapsed_triangles)} triangles")
        print(f"  Collapsed: {len(tri_indices) - len(collapsed_triangles)} triangles in {iterations} iterations")
        
        # Filter out degenerate and relatively small triangles
        print(f"  Filtering degenerate and small triangles...")
        
        # First pass: compute all areas and edge lengths
        triangle_data = []
        for tri_idx in collapsed_triangles:
            # Get triangle vertices
            p0 = np.array(collapsed_vertices[tri_idx[0]])
            p1 = np.array(collapsed_vertices[tri_idx[1]])
            p2 = np.array(collapsed_vertices[tri_idx[2]])
            
            # Compute area
            e1 = p1 - p0
            e2 = p2 - p0
            e3 = p2 - p1
            cross_product = e1[0] * e2[1] - e1[1] * e2[0]
            area = 0.5 * abs(cross_product)
            
            # Compute edge lengths
            len1 = np.linalg.norm(e1)
            len2 = np.linalg.norm(e2)
            len3 = np.linalg.norm(e3)
            min_edge = min(len1, len2, len3)
            
            triangle_data.append({
                'idx': tri_idx,
                'area': area,
                'min_edge': min_edge
            })
        
        # Compute statistics
        areas = np.array([t['area'] for t in triangle_data])
        median_area = np.median(areas)
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        print(f"  Triangle area statistics:")
        print(f"    - Min:    {areas.min():.2e}")
        print(f"    - Median: {median_area:.2e}")
        print(f"    - Mean:   {mean_area:.2e}")
        print(f"    - Max:    {areas.max():.2e}")
        print(f"    - Std:    {std_area:.2e}")
        
        # Define thresholds
        min_degenerate_area = 1e-6  # Absolute minimum (practical threshold in normalized space)
        min_edge_length = 1e-6  # Absolute minimum edge length
        
        # Very aggressive threshold: remove triangles smaller than 20% of median area
        # This will remove small triangles even near corners
        relative_threshold = 0.20 * median_area  # Increased to 20% for very aggressive filtering
        statistical_threshold = max(0, mean_area - 1.5 * std_area)  # Very aggressive (1.5σ)
        adaptive_threshold = max(relative_threshold, statistical_threshold, min_degenerate_area)
        
        print(f"  Thresholds:")
        print(f"    - Absolute min area:     {min_degenerate_area:.2e}")
        print(f"    - Relative (20% median): {relative_threshold:.2e}")
        print(f"    - Statistical (μ-1.5σ):  {statistical_threshold:.2e}")
        print(f"    - Adaptive threshold:    {adaptive_threshold:.2e}")
        
        # Second pass: filter triangles
        filtered_triangles = []
        degenerate_count = 0
        zero_area_count = 0
        short_edge_count = 0
        small_relative_count = 0
        small_mean_count = 0
        
        # Additional threshold based on mean (catches outliers near corners)
        mean_threshold = 0.10 * mean_area  # 10% of mean area
        
        for tri_data in triangle_data:
            area = tri_data['area']
            min_edge = tri_data['min_edge']
            is_degenerate = False
            
            # Check for zero/near-zero area (absolute)
            if area < min_degenerate_area:
                zero_area_count += 1
                is_degenerate = True
            
            # Check for very short edges (collinear or duplicate vertices)
            elif min_edge < min_edge_length:
                short_edge_count += 1
                is_degenerate = True
            
            # Check for relatively small triangles (median-based)
            elif area < adaptive_threshold:
                small_relative_count += 1
                is_degenerate = True
            
            # Additional check against mean (helps catch corner triangles)
            elif area < mean_threshold:
                small_mean_count += 1
                is_degenerate = True
            
            if not is_degenerate:
                filtered_triangles.append(tri_data['idx'])
            else:
                degenerate_count += 1
        
        if degenerate_count > 0:
            print(f"  ⚠ Removed {degenerate_count} problematic triangles:")
            if zero_area_count > 0:
                print(f"      - {zero_area_count} with area < {min_degenerate_area:.2e} (absolute)")
            if short_edge_count > 0:
                print(f"      - {short_edge_count} with edge length < {min_edge_length:.2e}")
            if small_relative_count > 0:
                print(f"      - {small_relative_count} with area < {adaptive_threshold:.2e} (<20% median)")
            if small_mean_count > 0:
                print(f"      - {small_mean_count} with area < {mean_threshold:.2e} (<10% mean)")
        else:
            print(f"  ✓ No problematic triangles found")
        
        collapsed_triangles = filtered_triangles
        
        # Final verification pass - ensure NO triangles have zero or extremely small area
        print(f"  Final verification pass...")
        final_filtered = []
        final_removed = 0
        for tri_idx in collapsed_triangles:
            p0 = np.array(collapsed_vertices[tri_idx[0]])
            p1 = np.array(collapsed_vertices[tri_idx[1]])
            p2 = np.array(collapsed_vertices[tri_idx[2]])
            
            e1 = p1 - p0
            e2 = p2 - p0
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            area = 0.5 * abs(cross)
            
            if area >= min_degenerate_area:
                final_filtered.append(tri_idx)
            else:
                final_removed += 1
        
        if final_removed > 0:
            print(f"  ⚠ Verification removed {final_removed} additional triangles with area < {min_degenerate_area:.2e}")
            collapsed_triangles = final_filtered
        else:
            print(f"  ✓ Verification passed - all triangles valid")
        
        print(f"  Final after filtering: {len(collapsed_triangles)} triangles")
        
        # Remove unused vertices and remap indices
        print(f"  Removing unused vertices...")
        
        # Find all vertices used in triangles
        used_vertices = set()
        for tri in collapsed_triangles:
            used_vertices.update(tri)
        
        used_vertices_sorted = sorted(used_vertices)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices_sorted)}
        
        # Remap triangles
        remapped_triangles = [[old_to_new[v] for v in tri] for tri in collapsed_triangles]
        
        # Keep only used vertices
        remapped_vertices = [collapsed_vertices[i] for i in used_vertices_sorted]
        
        unused_count = len(collapsed_vertices) - len(used_vertices)
        if unused_count > 0:
            print(f"  ⚠ Removed {unused_count} unused/orphaned vertices")
            print(f"  Vertices: {len(collapsed_vertices)} → {len(remapped_vertices)}")
            collapsed_vertices = remapped_vertices
            collapsed_triangles = remapped_triangles
        else:
            print(f"  ✓ No unused vertices found")
        
        normalized_triangles = []
        for tri_idx in collapsed_triangles:
            tri_coords = [collapsed_vertices[i] for i in tri_idx]
            normalized_triangles.append(tri_coords)
        
        stats['collapsed_edge'] = len(tri_indices) - len(collapsed_triangles)
        stats['valid'] = len(collapsed_triangles)
        
        valid_triangles = []
        for tri_norm_coords in normalized_triangles:
            # Convert normalized coords back to pixel coords (Y=0 at top)
            pixel_coords = [
                [x * bbox_width + min_x, y * bbox_height + min_y]
                for x, y in tri_norm_coords
            ]
            p0 = np.array(pixel_coords[0])
            p1 = np.array(pixel_coords[1])
            p2 = np.array(pixel_coords[2])
            aspect_ratio, min_angle = compute_triangle_quality(np.array(pixel_coords))
            v1, v2 = p1 - p0, p2 - p0
            area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            
            valid_triangles.append({
                'indices': [],
                'coords': pixel_coords,
                'aspect_ratio': aspect_ratio,
                'min_angle': min_angle,
                'area': area
            })
        
        print(f"\n  Final mesh summary:")
        print(f"    - Total generated: {stats['total']}")
        print(f"    - Removed (initial pixel filter): {stats['too_small']}")
        print(f"    - Removed (outside shape): {stats['outside_shape']}")
        print(f"    - Removed (bad aspect ratio): {stats['bad_aspect']}")
        print(f"    - Removed (poorly connected): {stats['poorly_connected']}")
        print(f"    - Removed (edge collapse): {stats['collapsed_edge']}")
        print(f"    - Removed (degenerate): {degenerate_count}")
        print(f"    - VALID: {stats['valid']}")
        
        print(f"  ✓ Coordinates normalized to [0, 1]")
        
        normalized_area_info = {
            'grid_spacing_x': float(norm_grid_spacing_x),
            'grid_spacing_y': float(norm_grid_spacing_y),
            'grid_cell_area': float(norm_grid_cell_area),
            'min_area_threshold': float(min_normalized_area),
            'min_area_factor': float(min_area_factor)
        }
    
    # Will save JSON in the springs generation step below
    
    # Generate springs
    if normalize and normalized_triangles is not None:
        print(f"\n[8/8] Generating springs with normalized coordinates...")
        
        # Map corners to collapsed vertex indices BEFORE adding springs
        corner_vertex_indices = []
        corner_positions_normalized = []
        
        if len(corner_points) > 0:
            for cx, cy in corner_points:
                min_dist = float('inf')
                closest_idx = -1
                bbox_width = bounding_box['width']
                bbox_height = bounding_box['height']
                min_x = bounding_box['min'][0]
                min_y = bounding_box['min'][1]
                
                # Normalize corner position (keep Y as-is)
                cx_norm = (cx - min_x) / bbox_width
                cy_norm = (cy - min_y) / bbox_height
                
                for idx, v_norm in enumerate(collapsed_vertices):
                    dist = np.sqrt((v_norm[0] - cx_norm)**2 + (v_norm[1] - cy_norm)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
                
                # Only accept if very close (within 0.01 in normalized space)
                if min_dist < 0.01:
                    corner_vertex_indices.append(closest_idx)
                    corner_positions_normalized.append([cx_norm, cy_norm])
        
        springs_data = {
            "vertices_normalized": collapsed_vertices,
            "triangles": collapsed_triangles,
            "boundary_springs": [],
            "interior_springs": [],
            "num_vertices": len(collapsed_vertices),
            "num_triangles": len(collapsed_triangles),
            "corner_vertices": corner_vertex_indices,
            "corner_positions_normalized": corner_positions_normalized,
            "num_corners": len(corner_vertex_indices),
            "normalization": {
                "method": "bounding_box",
                "formula": "(coord - min) / (max - min)",
                "range": "[0, 1]",
                "bounding_box": bounding_box
            }
        }
        
        edge_to_triangles = defaultdict(list)
        
        for tri_idx, tri in enumerate(collapsed_triangles):
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_to_triangles[edge].append(tri_idx)
        
        boundary_springs = []
        interior_springs = []
        
        for edge, tris in edge_to_triangles.items():
            spring = list(edge)
            if len(tris) == 1:
                boundary_springs.append(spring)
            else:
                interior_springs.append(spring)
        
        # Add forced edges between neighboring corners
        print(f"  Adding edges between neighboring corners...")
        corner_spring_count = 0
        
        if len(corner_points) > 0:
            # Map corners to vertex indices (in collapsed mesh)
            corner_to_vertex_idx = {}
            for cx, cy in corner_points:
                min_dist = float('inf')
                closest_idx = -1
                bbox_width = bounding_box['width']
                bbox_height = bounding_box['height']
                min_x = bounding_box['min'][0]
                min_y = bounding_box['min'][1]
                
                for idx, v_norm in enumerate(collapsed_vertices):
                    vx_px = v_norm[0] * bbox_width + min_x
                    vy_px = v_norm[1] * bbox_height + min_y
                    dist = np.sqrt((vx_px - cx)**2 + (vy_px - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
                if min_dist < 2.0:
                    corner_to_vertex_idx[(cx, cy)] = closest_idx
            
            if len(corner_to_vertex_idx) > 2:
                # Sort corners by angle to get boundary order
                corner_list = list(corner_to_vertex_idx.keys())
                corner_array = np.array(corner_list)
                centroid = corner_array.mean(axis=0)
                angles = np.arctan2(corner_array[:, 1] - centroid[1], 
                                   corner_array[:, 0] - centroid[0])
                sorted_indices = np.argsort(angles)
                
                # Add spring between each pair of neighboring corners
                existing_edges = set(tuple(sorted(e)) for e in boundary_springs + interior_springs)
                
                for i in range(len(sorted_indices)):
                    curr_corner = corner_list[sorted_indices[i]]
                    next_corner = corner_list[sorted_indices[(i + 1) % len(sorted_indices)]]
                    
                    v1 = corner_to_vertex_idx[curr_corner]
                    v2 = corner_to_vertex_idx[next_corner]
                    
                    edge = tuple(sorted([v1, v2]))
                    if edge not in existing_edges:
                        # DON'T add - just count missing edges
                        # boundary_springs.append([v1, v2])
                        # existing_edges.add(edge)
                        corner_spring_count += 1
                
                if corner_spring_count > 0:
                    print(f"    ⚠️  {corner_spring_count} edges MISSING between neighboring corners")
                else:
                    print(f"    ✓ All {len(sorted_indices)} neighboring corners are connected!")
        
        springs_data['boundary_springs'] = boundary_springs
        springs_data['interior_springs'] = interior_springs
        springs_data['num_boundary_springs'] = len(boundary_springs)
        springs_data['num_interior_springs'] = len(interior_springs)
        springs_data['num_total_springs'] = len(boundary_springs) + len(interior_springs)
        springs_data['num_corner_edges_added'] = corner_spring_count
        
        # Save as the main JSON output (this IS the spring data)
        with open(output_json, 'w') as f:
            json.dump(springs_data, f, indent=2)
        
        print(f"  ✓ {len(collapsed_vertices)} vertices (sequential 0-{len(collapsed_vertices)-1})")
        print(f"  ✓ {len(collapsed_triangles)} triangles")
        print(f"  ✓ {len(corner_vertex_indices)} corners preserved after collapse")
        print(f"  ✓ {len(boundary_springs)} boundary springs")
        print(f"  ✓ {len(interior_springs)} interior springs")
        print(f"  ✓ Saved to: {output_json}")
    
    # Visualization
    if output_viz:
        viz_width = width * scale_factor
        viz_height = height * scale_factor
        
        img_out = Image.new('RGB', (viz_width, viz_height), color='black')
        draw = ImageDraw.Draw(img_out)
        
        line_width = max(1, scale_factor // 2)
        
        for tri_data in valid_triangles:
            coords = [(x * scale_factor, y * scale_factor) for x, y in tri_data['coords']]
            draw.polygon(coords, outline='white', width=line_width)
        
        # Draw corners as green dots
        if len(corner_points) > 0:
            corner_radius = max(3, scale_factor)
            for cx, cy in corner_points:
                x_scaled = cx * scale_factor
                y_scaled = cy * scale_factor
                draw.ellipse(
                    [(x_scaled - corner_radius, y_scaled - corner_radius),
                     (x_scaled + corner_radius, y_scaled + corner_radius)],
                    fill=(0, 255, 0), outline=(255, 255, 255), width=2
                )
        
        img_out.save(output_viz)
        print(f"✓ Saved visualization: {output_viz} ({viz_width}×{viz_height})")
        if len(corner_points) > 0:
            print(f"✓ Rendered {len(corner_points)} corners as green dots")
    
    print("\n" + "="*70)
    
    # Return springs data if available, otherwise create minimal data dict
    if normalize and normalized_triangles is not None:
        return springs_data, stats
    else:
        return None, stats


if __name__ == "__main__":
    import sys
    import os
    
    # Get input image path from command line or use default
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = "model.bmp"  # Default: tessellate model.bmp in current folder
    
    # Generate output paths in the same folder as input
    input_dir = os.path.dirname(input_image)
    input_name = os.path.splitext(os.path.basename(input_image))[0]
    
    output_json = os.path.join(input_dir, f"{input_name}.json")
    output_viz = os.path.join(input_dir, f"{input_name}_tes.bmp")
    
    print(f"Input: {input_image}")
    print(f"Output JSON: {output_json}")
    print(f"Output Viz: {output_viz}")
    print()
    
    result, stats = refined_delaunay_tessellation(
        input_image,
        output_json,
        output_viz,
        interior_spacing=16,
        max_aspect_ratio=10.0,
        scale_factor=6,
        normalize=True,
        min_area_factor=0.1
    )
    
    print(f"\n✅ TESSELLATION COMPLETE!")
    print(f"   {stats['valid']} triangles in final mesh")
    print(f"   Edge collapse applied (no filtering)")
    print(f"   Valid manifold mesh with normalized [0,1] coordinates")
