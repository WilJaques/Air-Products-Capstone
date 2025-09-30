"""
Layout optimization module for flowsheet processing.

This module provides algorithms to optimize block positions to minimize
stream crossings and improve visual layout.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from models import Block, Stream
from collections import defaultdict

def optimize_block_positions(
    blocks: Dict[str, Block], 
    streams: Dict[str, Stream],
    iterations: int = 50,
    learning_rate: float = 0.1,
    preserve_layers: bool = True
) -> Dict[str, Block]:
    """
    Optimize block positions to minimize stream crossings and overlaps.
    
    Args:
        blocks: Dictionary of blocks with initial coordinates
        streams: Dictionary of streams with connectivity information
        iterations: Number of optimization iterations
        learning_rate: Step size for position updates
        preserve_layers: Whether to maintain layer structure
        
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
    
    # Group blocks into layers if preserving layers
    layers = []
    if preserve_layers:
        # Find roots (blocks with no incoming streams)
        roots = []
        for block_id, block in optimized_blocks.items():
            incoming = False
            for stream in streams.values():
                if block_id == stream.to_block:
                    incoming = True
                    break
            if not incoming:
                roots.append(block_id)
        
        # Build layers by traversing from roots
        visited = set()
        current_layer = roots
        
        while current_layer:
            layers.append(current_layer.copy())
            visited.update(current_layer)
            
            next_layer = []
            for block_id in current_layer:
                block = optimized_blocks[block_id]
                for output_stream_id in block.outputs:
                    if output_stream_id in streams:
                        stream = streams[output_stream_id]
                        if stream.to_block not in visited and stream.to_block in optimized_blocks:
                            next_layer.append(stream.to_block)
            
            current_layer = next_layer
    
    # Run optimization iterations
    for iteration in range(iterations):
        # Calculate forces between blocks
        forces = defaultdict(lambda: [0.0, 0.0])
        
        # Repulsive forces between all blocks
        for id1, block1 in optimized_blocks.items():
            for id2, block2 in optimized_blocks.items():
                if id1 != id2:
                    dx = block1.x_coord - block2.x_coord
                    dy = block1.y_coord - block2.y_coord
                    
                    # Avoid division by zero
                    distance = max(0.1, np.sqrt(dx*dx + dy*dy))
                    
                    # Repulsive force inversely proportional to distance
                    if distance < 3.0:  # Only consider nearby blocks
                        force = 1.0 / (distance * distance)
                        forces[id1][0] += force * dx / distance
                        forces[id1][1] += force * dy / distance
        
        # Attractive forces along streams
        for stream_id, stream in streams.items():
            if stream.from_block in optimized_blocks and stream.to_block in optimized_blocks:
                src_block = optimized_blocks[stream.from_block]
                dst_block = optimized_blocks[stream.to_block]
                
                dx = dst_block.x_coord - src_block.x_coord
                dy = dst_block.y_coord - src_block.y_coord
                
                distance = max(0.1, np.sqrt(dx*dx + dy*dy))
                
                # Attractive force proportional to distance
                if distance > 2.0:  # Only pull if blocks are far apart
                    force = 0.1 * distance
                    
                    forces[stream.from_block][0] += force * dx / distance
                    forces[stream.from_block][1] += force * dy / distance
                    
                    forces[stream.to_block][0] -= force * dx / distance
                    forces[stream.to_block][1] -= force * dy / distance
        
        # Update positions based on forces
        for block_id, block in optimized_blocks.items():
            # Apply forces with learning rate
            dx = learning_rate * forces[block_id][0]
            dy = learning_rate * forces[block_id][1]
            
            # Update positions
            block.x_coord += dx
            block.y_coord += dy
        
        # If preserving layers, adjust y-coordinates to maintain layers
        if preserve_layers:
            for i, layer in enumerate(layers):
                # Calculate average y-coordinate for this layer
                avg_y = sum(optimized_blocks[bid].y_coord for bid in layer) / len(layer)
                
                # Adjust y-coordinates to match layer position
                target_y = -5.0 - i * 3.0  # Start from top, move down
                delta_y = target_y - avg_y
                
                for block_id in layer:
                    optimized_blocks[block_id].y_coord += delta_y
            
            # For each layer, distribute blocks horizontally
            for layer in layers:
                # Sort blocks by current x-coordinate
                sorted_blocks = sorted(
                    [(bid, optimized_blocks[bid].x_coord) for bid in layer],
                    key=lambda x: x[1]
                )
                
                # Evenly distribute blocks horizontally
                min_x = min(optimized_blocks[bid].x_coord for bid in layer)
                max_x = max(optimized_blocks[bid].x_coord for bid in layer)
                
                if len(layer) > 1 and max_x > min_x:
                    for i, (block_id, _) in enumerate(sorted_blocks):
                        optimized_blocks[block_id].x_coord = min_x + i * (max_x - min_x) / (len(layer) - 1)
    
    return optimized_blocks