"""
Layout module for flowsheet processing.

This module contains functions for determining flowsheet layout, building
layers, and assigning coordinates to blocks and streams.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from models import Block

def find_first_elements(blocks: Dict[str, Block]) -> List[str]:
    """
    Find blocks that should be positioned first in the flowsheet layout.
    These are blocks whose inputs are not outputs of any other block.
    
    Args:
        blocks: Dictionary mapping block IDs to Block objects
        
    Returns:
        List of block IDs that should be positioned first
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
    return first_elements

def find_last_elements(blocks: Dict[str, Block]) -> List[str]:
    """
    Find blocks that should be positioned last in the flowsheet layout.
    These are blocks whose outputs are not inputs of any other block.
    
    Args:
        blocks: Dictionary mapping block IDs to Block objects
        
    Returns:
        List of block IDs that should be positioned last
    """
    # Collect all inputs from all blocks
    all_inputs = set()
    for block in blocks.values():
        all_inputs.update(block.inputs)

    # Find blocks where none of the outputs are in any inputs
    last_element = []
    for block in blocks.values():
        if block.outputs and all(out not in all_inputs for out in block.outputs):
            last_element.append(block.ID)
    return last_element

def build_index(blocks: Dict[str, Block]) -> Tuple[defaultdict, Dict[str, str]]:
    """
    Build lookup indices for stream-to-block relationships.
    
    Args:
        blocks: Dictionary mapping block IDs to Block objects
        
    Returns:
        Tuple containing:
        - input_to_blocks: Mapping from stream ID to list of blocks that use it as input
        - output_of: Mapping from stream ID to the block that produces it as output
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
    
    Args:
        blocks: Dictionary mapping block IDs to Block objects
        roots: List of block IDs to start the layering from
        
    Returns:
        List of layers, where each layer is a list of block IDs
    """
    input_to_blocks, _ = build_index(blocks)
    visited = set()
    layers = []
    current = list(dict.fromkeys(roots))
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
    else:
        # Add any last elements not already in the last layer to the last layer
        last_layer = set(layers[-1])
        for bid in last_elements:
            if bid not in last_layer:
                layers[-1].append(bid)

    # Optionally, add any blocks not yet assigned to any layer
    all_in_layers = {bid for layer in layers for bid in layer}
    remaining = [b for b in blocks if b not in all_in_layers]
    if remaining:
        layers.append(remaining)
    return layers

def assign_coordinates(layers: List[List[str]], dx: float = 10.0, dy: float = 6.0, spread: float = 1.0) -> Dict[str, Tuple[float, float]]:
    """
    Assign x,y coordinates: x by layer index, y spaced within each layer.
    
    Args:
        layers: List of layers, where each layer is a list of block IDs
        dx: Base horizontal spacing between layers
        dy: Base vertical spacing within layers
        spread: Global multiplier to spread everything out uniformly
        
    Returns:
        Dictionary mapping block IDs to (x, y) coordinate tuples
    """
    coords = {}
    for lx, layer in enumerate(layers):
        n = max(1, len(layer))
        # auto vertical spacing so tall layers don't get cramped
        layer_dy = dy * spread * max(1.0, min(1.0 + 0.12*(n-1), 2.0))
        y0 = -(n-1)/2.0 * layer_dy
        x = (dx * spread) * lx
        for i, bid in enumerate(layer):
            y = y0 + i * layer_dy
            coords[bid] = (x, y)
    return coords

def rescale_coords(coords: Dict[str, Tuple[float, float]], factor: float = 1.25) -> Dict[str, Tuple[float, float]]:
    """
    Uniformly scale all coordinates (quick 'zoom out').
    
    Args:
        coords: Dictionary mapping IDs to (x, y) coordinate tuples
        factor: Scaling factor to apply to all coordinates
        
    Returns:
        Dictionary with scaled coordinates
    """
    return {k: (x*factor, y*factor) for k, (x, y) in coords.items()}