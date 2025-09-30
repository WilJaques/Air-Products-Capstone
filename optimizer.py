"""
Layout optimization module for flowsheet processing.

This module provides algorithms to optimize block positions and stream routing
for clear, well-structured flowsheet diagrams.
"""

from typing import Dict, List, Tuple
import numpy as np
from models import Block, Stream, RoutePoint
from collections import defaultdict

def optimize_block_positions(
    blocks: Dict[str, Block], 
    streams: Dict[str, Stream],
    dx: float = 3.5,
    dy: float = 2.0,
    spread: float = 0.8,
    iterations: int = 50
) -> Dict[str, Block]:
    """
    Optimize block positions using a layered graph approach.
    
    Args:
        blocks: Dictionary of blocks with initial coordinates
        streams: Dictionary of streams with connectivity information
        dx: Horizontal spacing between layers
        dy: Vertical spacing within layers
        spread: Global spacing multiplier
        iterations: Number of optimization iterations (for fine-tuning)
        
    Returns:
        Dictionary of blocks with optimized coordinates
    """
    # Create a deep copy of blocks to avoid modifying the input
    optimized_blocks = {k: Block(
        ID=v.ID,
        inputs=v.inputs.copy() if v.inputs else [],
        outputs=v.outputs.copy() if v.outputs else [],
        x_coord=v.x_coord,
        y_coord=v.y_coord,
        block_type=v.block_type,
        icon=v.icon
    ) for k, v in blocks.items()}
    
    # Find starting blocks (roots)
    roots = find_first_elements(optimized_blocks)
    
    # If no clear roots found, use blocks with fewest inputs
    if not roots:
        input_counts = {block_id: len(block.inputs) for block_id, block in optimized_blocks.items()}
        min_inputs = min(input_counts.values()) if input_counts else 0
        roots = [block_id for block_id, count in input_counts.items() if count == min_inputs]
    
    # Build layers of blocks
    layers = build_layers(optimized_blocks, roots)
    
    # Assign coordinates based on layers
    coords = assign_coordinates(layers, dx=dx, dy=dy, spread=spread)
    
    # Update block coordinates
    for block_id, (x, y) in coords.items():
        optimized_blocks[block_id].x_coord = x
        optimized_blocks[block_id].y_coord = y
    
    # Fine-tune positions to avoid overlaps
    _fine_tune_positions(optimized_blocks, iterations=iterations)
    
    # Align blocks to grid for neat appearance
    _align_to_grid(optimized_blocks)
    
    return optimized_blocks

def find_first_elements(blocks: Dict[str, Block]) -> List[str]:
    """
    Find blocks that should be positioned first in the flowsheet layout.
    These are blocks whose inputs are not outputs of any other block.
    """
    # Collect all outputs from all blocks
    all_outputs = set()
    for block in blocks.values():
        all_outputs.update(block.outputs)

    # Find blocks where none of the inputs are in any outputs
    first_elements = []
    for block in blocks.values():
        if block.inputs and all(inp not in all_outputs for inp in block.inputs):
            first_elements.append(block.ID)
        # Also include blocks with no inputs
        elif not block.inputs and block.outputs:
            first_elements.append(block.ID)
    
    return first_elements

def find_last_elements(blocks: Dict[str, Block]) -> List[str]:
    """
    Find blocks that should be positioned last in the flowsheet layout.
    These are blocks whose outputs are not inputs of any other block.
    """
    # Collect all inputs from all blocks
    all_inputs = set()
    for block in blocks.values():
        all_inputs.update(block.inputs)

    # Find blocks where none of the outputs are in any inputs
    last_elements = []
    for block in blocks.values():
        if block.outputs and all(out not in all_inputs for out in block.outputs):
            last_elements.append(block.ID)
        # Also include blocks with no outputs
        elif not block.outputs and block.inputs:
            last_elements.append(block.ID)
    
    return last_elements

def build_index(blocks: Dict[str, Block]) -> Tuple[defaultdict, Dict[str, str]]:
    """
    Build lookup indices for stream-to-block relationships.
    """
    input_to_blocks = defaultdict(list)
    output_of = {}
    for b in blocks.values():
        for o in b.outputs:
            output_of[o] = b.ID
        for i in b.inputs:
            input_to_blocks[i].append(b.ID)
    return input_to_blocks, output_of

def build_layers(blocks: Dict[str, Block], roots: List[str]) -> List[List[str]]:
    """
    Build layers of blocks starting from roots (first elements).
    Ensures all blocks are included, and last elements are in the final layer.
    """
    input_to_blocks, _ = build_index(blocks)
    visited = set()
    layers = []
    current = list(dict.fromkeys(roots))
    
    # Handle empty roots case
    if not current and blocks:
        # Just use the first block as a starting point
        current = [next(iter(blocks.keys()))]
    
    while current:
        layers.append(current)
        nxt = []
        for bid in current:
            if bid in visited:
                continue
            visited.add(bid)
            for out in blocks[bid].outputs:
                for nb in input_to_blocks.get(out, []):
                    if nb not in visited and nb not in nxt:
                        nxt.append(nb)
        current = nxt

    # Ensure all last elements are in the final layer
    last_elements = find_last_elements(blocks)
    already_in_layers = {bid for layer in layers for bid in layer}
    missing_last = [bid for bid in last_elements if bid not in already_in_layers]
    if missing_last:
        layers.append(missing_last)
    
    # Add any blocks not yet assigned to any layer
    all_in_layers = {bid for layer in layers for bid in layer}
    remaining = [b for b in blocks if b not in all_in_layers]
    if remaining:
        layers.append(remaining)
    
    return layers

def assign_coordinates(layers: List[List[str]], dx: float = 5.0, dy: float = 3.0, spread: float = 1.0) -> Dict[str, Tuple[float, float]]:
    """
    Assign x,y coordinates: x by layer index, y spaced within each layer.
    """
    coords = {}
    for lx, layer in enumerate(layers):
        n = max(1, len(layer))
        # Auto vertical spacing so tall layers don't get cramped
        layer_dy = dy * spread * max(1.0, min(1.0 + 0.12*(n-1), 1.7))
        y0 = -(n-1)/2.0 * layer_dy
        x = (dx * spread) * lx
        for i, bid in enumerate(layer):
            y = y0 + i * layer_dy
            coords[bid] = (x, y)
    return coords

def _fine_tune_positions(blocks: Dict[str, Block], min_distance: float = 2.0, iterations: int = 1) -> None:
    """
    Apply fine-tuning to ensure blocks don't overlap.
    
    Args:
        blocks: Dictionary of blocks to adjust
        min_distance: Minimum distance to maintain between blocks
        iterations: Number of fine-tuning iterations
    """
    # Perform multiple iterations of fine-tuning if requested
    for _ in range(iterations):
        # Simple approach: slightly adjust block positions if they're too close
        block_list = list(blocks.values())
        for i in range(len(block_list)):
            for j in range(i+1, len(block_list)):
                b1, b2 = block_list[i], block_list[j]
                dx = b2.x_coord - b1.x_coord
                dy = b2.y_coord - b1.y_coord
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_distance:
                    # Push blocks apart slightly
                    push = (min_distance - dist) / 2
                    if abs(dx) > 0.01:  # Avoid division by zero
                        push_x = push * dx / abs(dx)
                        b1.x_coord -= push_x
                        b2.x_coord += push_x
                    
                    if abs(dy) > 0.01:  # Avoid division by zero
                        push_y = push * dy / abs(dy)
                        b1.y_coord -= push_y
                        b2.y_coord += push_y

def _align_to_grid(blocks: Dict[str, Block], grid_size: float = 0.5) -> None:
    """
    Align blocks to a grid for a neater layout.
    
    Args:
        blocks: Dictionary of blocks to align
        grid_size: Size of the grid cells
    """
    # Find common x and y coordinates to establish alignment patterns
    x_coords = defaultdict(list)
    y_coords = defaultdict(list)
    
    # Round to establish common coordinates
    for block_id, block in blocks.items():
        rounded_x = round(block.x_coord)
        rounded_y = round(block.y_coord)
        x_coords[rounded_x].append(block_id)
        y_coords[rounded_y].append(block_id)
    
    # Find groups of blocks that should be aligned (3 or more blocks with same coordinate)
    x_alignments = {x: ids for x, ids in x_coords.items() if len(ids) >= 2}
    y_alignments = {y: ids for y, ids in y_coords.items() if len(ids) >= 2}
    
    # Align blocks to these common coordinates
    for x, block_ids in x_alignments.items():
        for block_id in block_ids:
            # Move 70% toward the aligned position
            current_x = blocks[block_id].x_coord
            blocks[block_id].x_coord = current_x * 0.3 + x * 0.7
    
    for y, block_ids in y_alignments.items():
        for block_id in block_ids:
            # Move 70% toward the aligned position
            current_y = blocks[block_id].y_coord
            blocks[block_id].y_coord = current_y * 0.3 + y * 0.7
    
    # Final grid snap for all blocks
    for block in blocks.values():
        block.x_coord = round(block.x_coord / grid_size) * grid_size
        block.y_coord = round(block.y_coord / grid_size) * grid_size

def optimize_stream_routing(blocks: Dict[str, Block], streams: Dict[str, Stream]) -> Dict[str, List[RoutePoint]]:
    """
    Generate stream routes between blocks with improved terminal handling.
    
    Args:
        blocks: Dictionary of optimized block positions
        streams: Dictionary of streams with source and destination info
        
    Returns:
        Dictionary mapping stream IDs to lists of route points
    """
    # Identify terminal streams (inputs/outputs of the flowsheet)
    input_streams = []
    output_streams = []
    internal_streams = []
    
    # Build connection info for more informed stream routing
    connected_blocks = {}
    for stream_id, stream in streams.items():
        if stream.from_block and stream.to_block:
            if stream.from_block in blocks and stream.to_block in blocks:
                # This is an internal stream connecting two blocks
                internal_streams.append((stream_id, stream))
                connected_blocks.setdefault(stream.from_block, set()).add(stream.to_block)
                connected_blocks.setdefault(stream.to_block, set()).add(stream.from_block)
            elif stream.from_block in blocks:
                # This is an output terminal stream
                output_streams.append((stream_id, stream))
            elif stream.to_block in blocks:
                # This is an input terminal stream
                input_streams.append((stream_id, stream))
        elif stream.from_block in blocks:
            # Output terminal stream with no destination
            output_streams.append((stream_id, stream))
        elif stream.to_block in blocks:
            # Input terminal stream with no source
            input_streams.append((stream_id, stream))
    
    # Sort internal streams by distance (process shortest first)
    internal_streams.sort(key=lambda s: _stream_distance(s[1], blocks) if s[1].from_block in blocks and s[1].to_block in blocks else float('inf'))
    
    # Initialize result dictionary
    stream_routes = {}
    
    # Process internal streams first
    for stream_id, stream in internal_streams:
        src_block = blocks[stream.from_block]
        dst_block = blocks[stream.to_block]
        
        # Generate path with shorter routes
        path_points = _generate_shorter_path(src_block, dst_block)
        
        # Convert path to RoutePoints
        route_points = _convert_to_route_points(path_points)
        stream_routes[stream_id] = route_points
    
    # Process terminal streams - position them better
    
    # Input streams - position on left side or top of blocks
    for stream_id, stream in input_streams:
        if stream.to_block in blocks:
            dst_block = blocks[stream.to_block]
            
            # Position stream coming from left if possible
            external_point = (dst_block.x_coord - 2.0, dst_block.y_coord)
            
            # Generate shorter path
            path_points = [external_point, (dst_block.x_coord, dst_block.y_coord)]
            
            # Convert path to RoutePoints
            route_points = _convert_to_route_points(path_points)
            stream_routes[stream_id] = route_points
    
    # Output streams - position on right side or bottom of blocks
    for stream_id, stream in output_streams:
        if stream.from_block in blocks:
            src_block = blocks[stream.from_block]
            
            # Position stream going to right if possible
            external_point = (src_block.x_coord + 2.0, src_block.y_coord)
            
            # Generate shorter path
            path_points = [(src_block.x_coord, src_block.y_coord), external_point]
            
            # Convert path to RoutePoints
            route_points = _convert_to_route_points(path_points)
            stream_routes[stream_id] = route_points
    
    return stream_routes

def _stream_distance(stream: Stream, blocks: Dict[str, Block]) -> float:
    """Calculate direct distance between stream endpoints."""
    if stream.from_block not in blocks or stream.to_block not in blocks:
        return float('inf')
    
    src = blocks[stream.from_block]
    dst = blocks[stream.to_block]
    
    dx = dst.x_coord - src.x_coord
    dy = dst.y_coord - src.y_coord
    return np.sqrt(dx*dx + dy*dy)

def _generate_shorter_path(src_block: Block, dst_block: Block) -> List[Tuple[float, float]]:
    """Generate a shorter path between source and destination blocks."""
    # Start and end points
    start = (src_block.x_coord, src_block.y_coord)
    end = (dst_block.x_coord, dst_block.y_coord)
    
    # Calculate direct distance and block orientation
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # Check if blocks are close and directly aligned
    if abs(dx) < 0.1:  # Vertically aligned
        return [start, end]
    elif abs(dy) < 0.1:  # Horizontally aligned
        return [start, end]
    
    # For short distances, use a direct middle point
    if abs(dx) + abs(dy) < 5.0:
        # Use true midpoint for shorter path
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        return [start, (mid_x, mid_y), end]
    
    # For longer distances, use a more traditional routing
    if abs(dx) > abs(dy):
        # Primarily horizontal flow - use a direct bend in the middle
        mid_x = start[0] + dx * 0.5
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]
    else:
        # Primarily vertical flow - use a direct bend in the middle
        mid_y = start[1] + dy * 0.5
        return [start, (start[0], mid_y), (end[0], mid_y), end]

def _convert_to_route_points(path_points: List[Tuple[float, float]]) -> List[RoutePoint]:
    """Convert path points to RoutePoint objects for visualization."""
    route_points = []
    
    for i, (x, y) in enumerate(path_points):
        if i == 0:
            # Start point - determine direction based on next point
            next_x, next_y = path_points[1]
            direction = _determine_direction((x, y), (next_x, next_y))
            point = RoutePoint('r', direction, x, y, 0)
        elif i == len(path_points) - 1:
            # End point - determine direction based on previous point
            prev_x, prev_y = path_points[i-1]
            direction = _determine_direction((prev_x, prev_y), (x, y))
            point = RoutePoint('t', direction, x, y, 0)
        else:
            # Middle point - determine if it's a corner or straight segment
            prev_x, prev_y = path_points[i-1]
            
            if i < len(path_points) - 1:
                next_x, next_y = path_points[i+1]
                
                prev_horiz = abs(prev_y - y) < 0.001
                next_horiz = abs(next_y - y) < 0.001
                
                if prev_horiz != next_horiz:
                    # Direction changed - mark as a corner
                    point_type = 'x' if prev_horiz else 'y'
                    direction = point_type
                else:
                    # Continuing in same direction
                    point_type = 'x' if prev_horiz else 'y'
                    direction = point_type
            else:
                # Fall back for safety
                point_type = 'x'
                direction = 'x'
            
            point = RoutePoint(point_type, direction, x, y, 0)
        
        route_points.append(point)
    
    return route_points

def _determine_direction(from_point: Tuple[float, float], to_point: Tuple[float, float]) -> str:
    """Determine the direction from one point to another."""
    x1, y1 = from_point
    x2, y2 = to_point
    
    if abs(x2 - x1) > abs(y2 - y1):
        # Moving horizontally
        return 'r' if x2 > x1 else 'l'
    else:
        # Moving vertically
        return 'd' if y2 > y1 else 'u'