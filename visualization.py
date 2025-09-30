"""
Enhanced visualization module for flowsheet processing.

This module contains functions for drawing optimized flowsheets with
minimal stream crossings and overlaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple
from models import Block, Stream, Route, RoutePoint
import numpy as np

def draw_optimized_flowsheet(
    blocks: Dict[str, Block],
    streams: Dict[str, Stream],
    stream_paths: Dict[str, List[RoutePoint]],
    figsize: Tuple[int, int] = (16, 10),
    block_size: float = 1.0,
    output_path: str = None
) -> plt.Figure:
    """
    Draw a flowsheet with optimized block positions and stream paths.
    
    Args:
        blocks: Dictionary of blocks with coordinates
        streams: Dictionary of streams
        stream_paths: Dictionary of stream paths (lists of RoutePoints)
        figsize: Figure size (width, height) in inches
        block_size: Size of blocks in the visualization
        output_path: Path to save the figure (None for display only)
        
    Returns:
        Matplotlib Figure object
    """
    # Extract coordinates for bounds calculation
    x_coords = [block.x_coord for block in blocks.values()]
    y_coords = [block.y_coord for block in blocks.values()]
    
    # Add padding
    padding = 2.0
    x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
    y_min, y_max = min(y_coords) - padding, max(y_coords) + padding
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw streams first (so blocks appear on top)
    colors = plt.cm.Dark2.colors
    for i, (stream_id, path) in enumerate(stream_paths.items()):
        color = colors[i % len(colors)]
        
        # Extract points
        points = [(p.x, p.y) for p in path]
        
        # Draw segments
        for j in range(len(points) - 1):
            x1, y1 = points[j]
            x2, y2 = points[j+1]
            
            ax.plot([x1, x2], [y1, y2], 
                   color=color, 
                   linewidth=1.5, 
                   alpha=0.8,
                   solid_capstyle='round',
                   zorder=5)
        
        # Draw arrow at end
        if len(points) >= 2:
            # Get last two points for direction
            p1, p2 = points[-2], points[-1]
            
            # Calculate arrow direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                # Avoid zero direction
                dx, dy = 1.0, 0.0
                
            length = np.sqrt(dx*dx + dy*dy)
            dx, dy = dx/length, dy/length
            
            # Draw arrow
            ax.arrow(p2[0] - dx*0.5, p2[1] - dy*0.5,
                    dx*0.3, dy*0.3,
                    head_width=0.2,
                    head_length=0.3,
                    fc=color,
                    ec=color,
                    zorder=7)
        
        # Add stream label near middle of path
        if len(points) > 1:
            mid_idx = len(points) // 2
            mid_x = (points[mid_idx-1][0] + points[mid_idx][0]) / 2
            mid_y = (points[mid_idx-1][1] + points[mid_idx][1]) / 2
            
            ax.text(mid_x, mid_y, stream_id,
                   fontsize=8,
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                   zorder=20)
    
    # Draw blocks
    block_colors = {
        'HeatX': 'lightblue',
        'Flash2': 'lightgreen',
        'Compr': 'salmon',
        'Mixer': 'khaki',
        'RadFrac': 'plum',
        'Pump': 'paleturquoise',
        'RGibbs': 'orange',
        'RStoic': 'gold',
        'Valve': 'pink',
        'Sep': 'lightgray'
    }
    
    for block_id, block in blocks.items():
        # Get block type for color
        block_type = block.block_type if block.block_type else 'default'
        color = block_colors.get(block_type, 'lightgray')
        
        # Draw block as rectangle
        rect = patches.Rectangle(
            (block.x_coord - block_size/2, block.y_coord - block_size/2),
            block_size, block_size,
            facecolor=color,
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            zorder=10
        )
        ax.add_patch(rect)
        
        # Add block label
        ax.text(block.x_coord, block.y_coord, block_id,
               fontsize=9,
               ha='center', va='center',
               fontweight='bold',
               zorder=15)
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add title
    ax.set_title('Optimized Process Flowsheet')
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig