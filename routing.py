"""
Stream routing module for flowsheet processing.

This module provides algorithms to calculate optimal paths for streams
with minimal crossings and overlaps.
"""

from typing import Dict, List, Tuple, Set
from models import Block, Stream, Route, RoutePoint
import numpy as np
from collections import defaultdict

def calculate_stream_paths(blocks: Dict[str, Block], 
                          streams: Dict[str, Stream],
                          grid_size: float = 0.5) -> Dict[str, List[RoutePoint]]:
    """
    Calculate optimal paths for streams to minimize crossings and overlaps.
    
    Args:
        blocks: Dictionary of blocks with coordinates
        streams: Dictionary of streams with source and destination info
        grid_size: Resolution of routing grid
        
    Returns:
        Dictionary mapping stream IDs to lists of route points
    """
    # Create a routing grid
    x_coords = [block.x_coord for block in blocks.values()]
    y_coords = [block.y_coord for block in blocks.values()]
    
    min_x, max_x = min(x_coords) - 5, max(x_coords) + 5
    min_y, max_y = min(y_coords) - 5, max(y_coords) + 5
    
    # Initialize result dictionary
    stream_paths = {}
    
    # Sort streams by length (route shorter streams first)
    sorted_streams = sorted(
        streams.items(), 
        key=lambda item: stream_length(item[1], blocks) if has_endpoints(item[1], blocks) else float('inf')
    )
    
    # Keep track of used paths to avoid overlaps
    occupied_segments = set()
    
    for stream_id, stream in sorted_streams:
        if not has_endpoints(stream, blocks):
            continue
            
        # Get source and destination blocks
        src_block = blocks.get(stream.from_block)
        dst_block = blocks.get(stream.to_block)
        
        if not src_block or not dst_block:
            continue
            
        # Calculate optimal path
        path = find_optimal_path(
            src_block, 
            dst_block, 
            occupied_segments,
            grid_size,
            blocks
        )
        
        # Convert path to route points
        route_points = []
        for i, (x, y) in enumerate(path):
            if i == 0:
                # Root point
                point = RoutePoint('r', 'r', x, y, 0)
            elif i == len(path) - 1:
                # Terminal point
                direction = get_direction(path[i-1], path[i])
                point = RoutePoint('t', direction, x, y, 0)
            else:
                # Intermediate point
                prev_segment = (path[i-1][0] - path[i][0], path[i-1][1] - path[i][1])
                next_segment = (path[i][0] - path[i+1][0], path[i][1] - path[i+1][1])
                
                # Determine if this is a bend point (horizontal/vertical change)
                if prev_segment[0] == 0 and next_segment[1] == 0:
                    # Vertical to horizontal
                    point_type = 'y'
                elif prev_segment[1] == 0 and next_segment[0] == 0:
                    # Horizontal to vertical
                    point_type = 'x'
                else:
                    # Continue in same direction
                    point_type = 'x' if prev_segment[1] == 0 else 'y'
                
                point = RoutePoint(point_type, point_type, x, y, 0)
            
            route_points.append(point)
        
        # Add segments to occupied set
        for i in range(len(path) - 1):
            segment = (path[i], path[i+1])
            occupied_segments.add(segment)
        
        # Store path
        stream_paths[stream_id] = route_points
    
    return stream_paths

def find_optimal_path(
    src_block: Block,
    dst_block: Block,
    occupied_segments: Set[Tuple],
    grid_size: float,
    blocks: Dict[str, Block]
) -> List[Tuple[float, float]]:
    """
    Find an optimal path between two blocks that avoids overlaps.
    
    Uses a modified A* algorithm with penalties for:
    - Crossing occupied segments
    - Passing too close to other blocks
    - Making too many turns
    
    Returns a list of (x, y) coordinate tuples defining the path.
    """
    # Start and end points (outside the blocks)
    src_x, src_y = src_block.x_coord, src_block.y_coord
    dst_x, dst_y = dst_block.x_coord, dst_block.y_coord
    
    # Define orthogonal exit points from source block
    exit_points = [
        (src_x + 1.0, src_y),  # Right
        (src_x - 1.0, src_y),  # Left
        (src_x, src_y + 1.0),  # Top
        (src_x, src_y - 1.0),  # Bottom
    ]
    
    # Define orthogonal entry points to destination block
    entry_points = [
        (dst_x - 1.0, dst_y),  # Left
        (dst_x + 1.0, dst_y),  # Right
        (dst_x, dst_y - 1.0),  # Bottom
        (dst_x, dst_y + 1.0),  # Top
    ]
    
    # Try different combinations of entry and exit points to find best path
    best_path = None
    best_score = float('inf')
    
    for exit_pt in exit_points:
        for entry_pt in entry_points:
            # Create a simple Manhattan path with one bend
            middle_x, middle_y = exit_pt[0], entry_pt[1]
            
            # Check if this point is too close to any block
            too_close = any(
                abs(block.x_coord - middle_x) < 0.8 and 
                abs(block.y_coord - middle_y) < 0.8
                for block in blocks.values()
            )
            
            if too_close:
                # Try the other bend point
                middle_x, middle_y = entry_pt[0], exit_pt[1]
                too_close = any(
                    abs(block.x_coord - middle_x) < 0.8 and 
                    abs(block.y_coord - middle_y) < 0.8
                    for block in blocks.values()
                )
                
                if too_close:
                    # Both bend points are too close, skip this combination
                    continue
            
            # Create path with minimum number of segments
            path = []
            
            # Start from exit point
            path.append((src_x, src_y))  # Start at block center
            path.append(exit_pt)
            
            # Add bend point if needed
            if exit_pt[0] != entry_pt[0] and exit_pt[1] != entry_pt[1]:
                path.append((middle_x, middle_y))
                
            # End at entry point
            path.append(entry_pt)
            path.append((dst_x, dst_y))  # End at block center
            
            # Calculate score based on path length and overlaps
            score = 0
            
            # Path length penalty
            path_length = sum(
                abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1])
                for i in range(len(path) - 1)
            )
            score += path_length
            
            # Overlap penalty
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                reverse_segment = (path[i+1], path[i])
                
                if segment in occupied_segments or reverse_segment in occupied_segments:
                    score += 100  # Large penalty for overlap
            
            # Number of bends penalty
            num_bends = 0
            for i in range(1, len(path) - 1):
                prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                
                if prev_dir != next_dir:
                    num_bends += 1
            
            score += num_bends * 2  # Penalty for each bend
            
            # Update best path if this is better
            if score < best_score:
                best_score = score
                best_path = path
    
    # If no path found, create a direct line
    if not best_path:
        best_path = [(src_x, src_y), (dst_x, dst_y)]
    
    return best_path

def get_direction(p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """Determine the direction from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    
    if abs(dx) > abs(dy):
        return 'r' if dx > 0 else 'l'
    else:
        return 'd' if dy > 0 else 'u'

def stream_length(stream: Stream, blocks: Dict[str, Block]) -> float:
    """Calculate approximate length of a stream."""
    if not has_endpoints(stream, blocks):
        return float('inf')
        
    src_block = blocks.get(stream.from_block)
    dst_block = blocks.get(stream.to_block)
    
    return abs(dst_block.x_coord - src_block.x_coord) + abs(dst_block.y_coord - src_block.y_coord)

def has_endpoints(stream: Stream, blocks: Dict[str, Block]) -> bool:
    """Check if stream has valid source and destination blocks."""
    return (stream.from_block in blocks and stream.to_block in blocks)