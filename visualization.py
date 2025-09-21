"""
Visualization module for flowsheet processing.

This module contains functions for drawing flowsheets and calculating
stream coordinates for visualization.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from models import Block, Stream
from layout import build_index

def stream_coords(streams: Dict[str, Stream], block_coords: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[List[Tuple[float, float]], Tuple[float, float]]]:
    """
    Creates 90-degree angle paths for stream connections with right angle turns.
    
    Args:
        streams: Dictionary mapping stream IDs to Stream objects
        block_coords: Dictionary mapping block IDs to (x, y) coordinate tuples
        
    Returns:
        Dictionary mapping stream IDs to (path_points, label_position) tuples
    """
    pair_to_streams = defaultdict(list)
    for s in streams.values():
        if s.from_block in block_coords and s.to_block in block_coords:
            pair = (s.from_block, s.to_block)
            pair_to_streams[pair].append(s.ID)

    coords = {}
    offset_dist = 0.8  # Base offset distance from block center
    block_radius = 0.5  # Estimated visual size of block to avoid overlap

    for (from_block, to_block), stream_ids in pair_to_streams.items():
        x1, y1 = block_coords[from_block]
        x2, y2 = block_coords[to_block]
        n = len(stream_ids)
        
        # Calculate perpendicular direction for offsets
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5 or 1.0
        perp_x = -dy / length
        perp_y = dx / length
        
        for i, stream_id in enumerate(stream_ids):
            # Calculate offset for this specific stream
            stream_offset = (i - (n-1)/2) * offset_dist
            
            # Starting point - offset from first block
            sx = x1 + perp_x * stream_offset
            sy = y1 + perp_y * stream_offset
            
            # Exit point from block (with clearance)
            exit_x = x1 + (dx / length) * block_radius
            exit_y = y1 + (dy / length) * block_radius
            
            # Ending point - offset from second block
            ex = x2 + perp_x * stream_offset
            ey = y2 + perp_y * stream_offset
            
            # Entry point to block (with clearance)
            entry_x = x2 - (dx / length) * block_radius
            entry_y = y2 - (dy / length) * block_radius
            
            # Create path with 90-degree angles
            # First determine if primarily horizontal or vertical
            if abs(dx) > abs(dy):  # Horizontal dominant
                # Create horizontal-vertical-horizontal path
                mid_x = (sx + ex) / 2
                path = [
                    (sx, sy),
                    (mid_x, sy),
                    (mid_x, ey),
                    (ex, ey)
                ]
                # Label position at the vertical segment
                label_pos = (mid_x, (sy + ey) / 2)
            else:  # Vertical dominant
                # Create vertical-horizontal-vertical path
                mid_y = (sy + ey) / 2
                path = [
                    (sx, sy),
                    (sx, mid_y),
                    (ex, mid_y),
                    (ex, ey)
                ]
                # Label position at the horizontal segment
                label_pos = ((sx + ex) / 2, mid_y)
                
            coords[stream_id] = (path, label_pos)
    return coords

def draw_flowsheet(blocks: Dict[str, Block], coords: Dict[str, Tuple[float, float]], 
                  streams: Optional[Dict[str, Stream]] = None, 
                  stream_coords_dict: Optional[Dict[str, Tuple[List[Tuple[float, float]], Tuple[float, float]]]] = None, 
                  figsize: Tuple[int, int] = (14, 8), margin: float = 6.0) -> None:
    """
    Draw the flowsheet with blocks and streams.
    
    Args:
        blocks: Dictionary mapping block IDs to Block objects
        coords: Dictionary mapping block IDs to (x, y) coordinate tuples
        streams: Optional dictionary mapping stream IDs to Stream objects
        stream_coords_dict: Optional dictionary with stream paths and label positions
        figsize: Figure size as (width, height) tuple
        margin: Margin around the plot
    """
    input_to_blocks, _ = build_index(blocks)
    xs, ys = zip(*coords.values())
    xmin, xmax = min(xs)-margin, max(xs)+margin
    ymin, ymax = min(ys)-margin, max(ys)+margin

    fig, ax = plt.subplots(figsize=figsize)

    # Draw blocks
    for bid, (x, y) in coords.items():
        ax.scatter([x], [y], s=160, color='skyblue', zorder=3)
        ax.text(x, y+0.7, bid, ha='center', va='bottom', fontsize=9, zorder=4)

    # Draw streams with 90-degree angles
    if streams and stream_coords_dict:
        for s in streams.values():
            if s.ID in stream_coords_dict:
                path, label_pos = stream_coords_dict[s.ID]
                
                # Draw the path segments
                for i in range(len(path)-1):
                    (x1, y1), (x2, y2) = path[i], path[i+1]
                    ax.plot([x1, x2], [y1, y2], color='orange', lw=1.5, zorder=1)
                
                # Add arrow at the end
                end_segment = path[-2], path[-1]
                mid_x = (end_segment[0][0] + end_segment[1][0]) / 2
                mid_y = (end_segment[0][1] + end_segment[1][1]) / 2
                dx = end_segment[1][0] - end_segment[0][0]
                dy = end_segment[1][1] - end_segment[0][1]
                
                # Create arrow
                ax.annotate("", 
                    xy=end_segment[1], xytext=(mid_x, mid_y),
                    arrowprops=dict(arrowstyle="->", color='orange', lw=1.5),
                    zorder=2
                )
                
                # Add stream label
                lx, ly = label_pos
                ax.text(lx, ly, s.ID, color='orange', fontsize=8, 
                        ha='center', va='center', zorder=2,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.set_title("Auto-arranged flowsheet")
    ax.set_xlabel("Layer (left → right)")
    ax.set_ylabel("Relative position")
    plt.show()