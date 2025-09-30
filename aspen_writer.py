"""
Aspen input file writer module with proper route formatting.

This module provides functions to update an Aspen input file with optimized
block positions and stream routes while ensuring correct route structure.
"""

from typing import Dict, List, Tuple
import re
from models import Block, Stream, RoutePoint

def update_aspen_file(
    input_file: str,
    output_file: str,
    blocks: Dict[str, Block],
    stream_paths: Dict[str, List[RoutePoint]]
) -> None:
    """
    Update an Aspen input file with optimized block and stream positions.
    
    Args:
        input_file: Path to original Aspen input file
        output_file: Path to save updated file
        blocks: Dictionary of blocks with optimized coordinates
        stream_paths: Dictionary of optimized stream paths
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Update block coordinates
    for block_id, block in blocks.items():
        # Match the block section and its coordinates
        block_pattern = f";ID: {re.escape(block_id)}\\s*\\n;Version: \\d+\\s*\\n;ICON:.*?\\n;Flag.*?\\n;Section.*?\\n;At ([\\d.-]+) ([\\d.-]+)"
        match = re.search(block_pattern, content)
        
        if match:
            # Replace with new coordinates
            old_coords = f";At {match.group(1)} {match.group(2)}"
            new_coords = f";At {block.x_coord:.6f} {block.y_coord:.6f}"
            content = content.replace(old_coords, new_coords)
    
    # Update stream routes
    for stream_id, route_points in stream_paths.items():
        # Match the stream section
        stream_pattern = f";ID: {re.escape(stream_id)}\\s*\\n.*?\\n.*?\\n.*?\\n;At.*?\\n.*?\\n;ROUTE (\\d+) (\\d+)"
        
        # Find start of the route section
        match = re.search(stream_pattern, content, re.DOTALL)
        
        if match:
            route_num = match.group(1)
            route_idx = match.group(2)
            route_header = f";ROUTE {route_num} {route_idx}"
            
            # Find the position of the route header
            route_pos = content.find(route_header, match.start())
            if route_pos != -1:
                # Find where to start replacing
                route_start = route_pos + len(route_header)
                
                # Find where to end replacing (next ROUTE or STREAM or At)
                next_section_pattern = r"(;ROUTE|\$ROUTE|;STREAM|;At)"
                next_section = re.search(next_section_pattern, content[route_start:])
                if next_section:
                    route_end = route_start + next_section.start()
                else:
                    # If no next section found, just go to the next line that starts with ;
                    next_line = content[route_start:].find("\n;")
                    if next_line != -1:
                        route_end = route_start + next_line
                    else:
                        route_end = len(content)
                
                # Generate new route string with proper formatting
                new_route = "\n"  # Start with a newline
                
                # Add route points if any
                if route_points:
                    # First point is always 'r' (root)
                    first_point = route_points[0]
                    direction = first_point.direction
                    if direction not in ['r', 'l', 'u', 'd']:
                        direction = 'r'  # Default for root
                    new_route += f";r {direction} {first_point.x:.6f} {first_point.y:.6f} {first_point.z}\n"
                    
                    # Middle points (if any) - ensure proper type and direction
                    for i in range(1, len(route_points) - 1):
                        point = route_points[i]
                        point_type = point.point_type
                        
                        # For x/y points, make sure direction matches type
                        if point_type in ['x', 'y']:
                            direction = point_type  # Direction should match type for x/y
                        else:
                            direction = point.direction
                            if direction not in ['r', 'l', 'u', 'd', 'x', 'y']:
                                direction = 'x'  # Default
                                
                        new_route += f";{point_type} {direction} {point.x:.6f} {point.y:.6f} {point.z}\n"
                    
                    # Last point is terminal - ensure proper direction
                    if len(route_points) > 1:
                        last_point = route_points[-1]
                        prev_point = route_points[-2]
                        
                        # Calculate terminal direction based on geometry
                        direction = calculate_terminal_direction(prev_point, last_point)
                        new_route += f";t {direction} {last_point.x:.6f} {last_point.y:.6f} {last_point.z}\n"
                
                # Add closing dollar sign lines with proper format
                new_route += ";$ C 1.000000 0.0\n"
                new_route += ";$ $ 0.0 0.0"
                
                # Replace old route with new one
                content = content[:route_start] + new_route + content[route_end:]
    
    # CRITICAL FIX: Split any lines with both dollar sign and ROUTE declarations
    # This specifically fixes patterns like ";$ $ 0.0 0.0;ROUTE 1 0"
    content = re.sub(r'(;\$ \$ 0\.0 0\.0);(ROUTE \d+ \d+)', r'\1\n;\2', content)
    
    # Also fix any other cases where "ROUTE" appears on the same line after another command
    content = re.sub(r'([^\n]);(ROUTE \d+ \d+)', r'\1\n;\2', content)
    
    # Write updated content to output file
    with open(output_file, 'w') as f:
        f.write(content)

def calculate_terminal_direction(prev_point: RoutePoint, terminal_point: RoutePoint) -> str:
    """
    Calculate the correct terminal direction based on geometry.
    
    Args:
        prev_point: The point before the terminal point
        terminal_point: The terminal point
        
    Returns:
        Direction code ('r', 'l', 'u', or 'd')
    """
    dx = terminal_point.x - prev_point.x
    dy = terminal_point.y - prev_point.y
    
    # Determine dominant direction
    if abs(dx) > abs(dy):
        # Horizontal movement dominates
        return 'r' if dx > 0 else 'l'
    else:
        # Vertical movement dominates
        return 'd' if dy > 0 else 'u'

def generate_route_string(route_points, route_num=0, route_idx=0):
    """
    Generate a properly formatted Aspen route string.
    
    Args:
        route_points: List of RoutePoint objects
        route_num: Route number
        route_idx: Route index
        
    Returns:
        String in Aspen route format
    """
    result = f";ROUTE {route_num} {route_idx}\n"
    
    # Add route points
    if not route_points:
        # Empty route
        result += ";$ $ 0.0 0.0"
        return result
    
    # First point is root
    first_point = route_points[0]
    direction = first_point.direction if first_point.direction in ['r', 'l', 'u', 'd'] else 'r'
    result += f";r {direction} {first_point.x:.6f} {first_point.y:.6f} {first_point.z}\n"
    
    # Middle points
    for i in range(1, len(route_points) - 1):
        point = route_points[i]
        point_type = point.point_type
        direction = point_type if point_type in ['x', 'y'] else point.direction
        result += f";{point_type} {direction} {point.x:.6f} {point.y:.6f} {point.z}\n"
    
    # Last point is terminal
    if len(route_points) > 1:
        last_point = route_points[-1]
        prev_point = route_points[-2]
        direction = calculate_terminal_direction(prev_point, last_point)
        result += f";t {direction} {last_point.x:.6f} {last_point.y:.6f} {last_point.z}\n"
    
    # Add ending dollar sign lines - on separate lines from any route declaration
    result += ";$ C 1.000000 0.0\n";
    result += ";$ $ 0.0 0.0";
    
    return result