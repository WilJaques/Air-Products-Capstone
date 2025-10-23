"""
Stream routing module for flowsheet processing.

This module provides algorithms to calculate optimal paths for streams
with minimal crossings and overlaps.
"""

from typing import Dict, List, Tuple
from models import Block, Stream, RoutePoint

def calculate_stream_paths(blocks: Dict[str, Block], streams: Dict[str, Stream]) -> Dict[str, List[RoutePoint]]:
    """
    Calculate optimal stream paths between blocks using orthogonal routing.
    
    Args:
        blocks: Dictionary of block objects with optimized positions
        streams: Dictionary of stream objects
        
    Returns:
        Dictionary mapping stream IDs to lists of RoutePoint objects
    """
    stream_paths = {}
    
    for stream_id, stream in streams.items():
        from_block = blocks.get(stream.from_block)
        to_block = blocks.get(stream.to_block)
        
        if not from_block or not to_block:
            continue
        
        # Get start and end coordinates
        x1, y1 = from_block.x_coord, from_block.y_coord
        x2, y2 = to_block.x_coord, to_block.y_coord
        
        # Create route points
        route_points = []
        
        # Start point (root)
        start_direction = _determine_initial_direction(x1, y1, x2, y2)
        route_points.append(RoutePoint(
            x=x1, y=y1, z=0,
            point_type='r',
            direction=start_direction
        ))
        
        # Middle waypoints for orthogonal routing
        if abs(x2 - x1) > 0.1 and abs(y2 - y1) > 0.1:
            # L-shaped or Z-shaped path needed
            mid_x = (x1 + x2) / 2
            
            # First turn
            route_points.append(RoutePoint(
                x=mid_x, y=y1, z=0,
                point_type='x',
                direction='x'
            ))
            
            # Second turn
            route_points.append(RoutePoint(
                x=mid_x, y=y2, z=0,
                point_type='y',
                direction='y'
            ))
        elif abs(y2 - y1) > 0.1:
            # Only vertical movement needed
            route_points.append(RoutePoint(
                x=x1, y=y2, z=0,
                point_type='y',
                direction='y'
            ))
        elif abs(x2 - x1) > 0.1:
            # Only horizontal movement needed
            route_points.append(RoutePoint(
                x=x2, y=y1, z=0,
                point_type='x',
                direction='x'
            ))
        
        # End point (terminal)
        end_direction = _determine_terminal_direction(route_points[-1], x2, y2)
        route_points.append(RoutePoint(
            x=x2, y=y2, z=0,
            point_type='t',
            direction=end_direction
        ))
        
        stream_paths[stream_id] = route_points
    
    return stream_paths


def _determine_initial_direction(x1: float, y1: float, x2: float, y2: float) -> str:
    """Determine initial direction from start to end point."""
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) > abs(dy):
        return 'r' if dx > 0 else 'l'
    else:
        return 'd' if dy > 0 else 'u'


def _determine_terminal_direction(last_point: RoutePoint, x2: float, y2: float) -> str:
    """Determine terminal direction based on last waypoint."""
    dx = x2 - last_point.x
    dy = y2 - last_point.y
    
    if abs(dx) > abs(dy):
        return 'r' if dx > 0 else 'l'
    else:
        return 'd' if dy > 0 else 'u'