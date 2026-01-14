# SPDX-FileCopyrightText: Copyright (c) 2025 NBEL
# SPDX-License-Identifier: Apache-2.0
#
# Graph coloring utilities for 2D VBD solver
# Enables parallel vertex updates by grouping non-adjacent vertices

import numpy as np


def compute_adjacency_from_triangles(tri_indices: np.ndarray, num_vertices: int) -> tuple:
    """
    Compute vertex adjacency from triangle connectivity.
    
    Two vertices are adjacent if they share a triangle edge.
    This determines which vertices can be updated in parallel.
    
    Args:
        tri_indices: Flattened triangle indices [i0, j0, k0, i1, j1, k1, ...]
        num_vertices: Total number of vertices
    
    Returns:
        adjacency: Dictionary mapping vertex -> set of adjacent vertices
        max_degree: Maximum number of neighbors for any vertex
    """
    num_triangles = len(tri_indices) // 3
    adjacency = {i: set() for i in range(num_vertices)}
    max_degree = 0
    
    for t in range(num_triangles):
        # Get triangle vertices
        i = tri_indices[t * 3 + 0]
        j = tri_indices[t * 3 + 1]
        k = tri_indices[t * 3 + 2]
        
        # Add bidirectional edges
        adjacency[i].add(j)
        adjacency[i].add(k)
        adjacency[j].add(i)
        adjacency[j].add(k)
        adjacency[k].add(i)
        adjacency[k].add(j)
        
        # Update max degree
        max_degree = max(max_degree, len(adjacency[i]), len(adjacency[j]), len(adjacency[k]))
    
    return adjacency, max_degree


def compute_vertex_to_triangle_adjacency(tri_indices: np.ndarray, num_vertices: int) -> tuple:
    """
    Build mapping from vertex to incident triangles.
    
    Used during VBD iteration to find all triangles contributing
    forces/Hessians to a given vertex.
    
    Args:
        tri_indices: Flattened triangle indices
        num_vertices: Total number of vertices
    
    Returns:
        adj_v2t: numpy array [num_vertices, max_incident], padded with -1
        max_incident: Maximum number of triangles incident to any vertex
    """
    num_triangles = len(tri_indices) // 3
    
    # Build list of incident triangles for each vertex
    incident_lists = [[] for _ in range(num_vertices)]
    max_incident = 0
    
    for t in range(num_triangles):
        i = tri_indices[t * 3 + 0]
        j = tri_indices[t * 3 + 1]
        k = tri_indices[t * 3 + 2]
        
        incident_lists[i].append(t)
        incident_lists[j].append(t)
        incident_lists[k].append(t)
        
        max_incident = max(max_incident, 
                          len(incident_lists[i]),
                          len(incident_lists[j]),
                          len(incident_lists[k]))
    
    # Convert to fixed-size numpy array
    adj_v2t = np.full((num_vertices, max_incident), -1, dtype=np.int32)
    for v in range(num_vertices):
        for idx, t in enumerate(incident_lists[v]):
            adj_v2t[v, idx] = t
    
    return adj_v2t, max_incident


def graph_coloring_2d(adjacency: dict) -> tuple:
    """
    Greedy graph coloring for VBD parallelization.
    
    Assigns colors to vertices such that no two adjacent vertices
    share the same color. This allows all vertices of the same color
    to be updated in parallel without race conditions.
    
    Algorithm: Simple greedy coloring (sufficient for mesh graphs).
    For 2D triangular meshes, typically requires 3-7 colors.
    
    Args:
        adjacency: Dictionary mapping vertex -> set of adjacent vertices
    
    Returns:
        coloring: Array of color assignments per vertex
        color_groups: Dictionary mapping color -> list of vertices
    
    Theory (VBD Paper):
    ------------------
    VBD exploits the local nature of FEM by solving small independent
    linear systems per vertex. Graph coloring partitions vertices such
    that non-interfering updates can be batched.
    
    The number of colors scales with mesh degree, not mesh size,
    enabling near-linear scaling on GPUs.
    """
    MAX_COLORS = 256
    num_vertices = len(adjacency)
    coloring = -1 * np.ones(num_vertices, dtype=np.int32)
    color_groups = {}
    
    for vertex in range(num_vertices):
        # Find colors used by neighbors
        used_colors = set()
        for neighbor in adjacency[vertex]:
            if coloring[neighbor] != -1:
                used_colors.add(coloring[neighbor])
        
        # Assign lowest available color
        for color in range(MAX_COLORS):
            if color not in used_colors:
                coloring[vertex] = color
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(vertex)
                break
    
    num_colors = max(coloring) + 1
    color_distribution = np.bincount(coloring)
    print(f"  Graph coloring: {num_colors} colors, distribution: {color_distribution}")
    
    return coloring, color_groups
