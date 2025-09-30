"""
Enhanced parsing module for flowsheet processing.

This module contains functions for parsing Aspen Plus input files, extracting
block and stream information including route coordinates for visualization.
"""

import re
from typing import Dict, List, Tuple
from models import Block, Stream, Route, RoutePoint

# Regular expressions for PFS section elements
BLOCK_SECTION_RE = re.compile(r';BLOCK\s*\n;ID:\s*([A-Za-z0-9_]+)')
STREAM_SECTION_RE = re.compile(r';STREAM\s*\n;ID:\s*([A-Za-z0-9_]+)')
AT_COORD_RE = re.compile(r';At\s+([-\d\.]+)\s+([-\d\.]+)')
ROUTE_START_RE = re.compile(r';ROUTE\s+(\d+)\s+(\d+)')
ROUTE_POINT_RE = re.compile(r';([rtxy])\s+([rludxy])\s+([-\d\.]+)\s+([-\d\.]+)\s+(\d+)')

def parse_flowsheet_pfs(text: str) -> Tuple[Dict[str, Block], Dict[str, Stream]]:
    """
    Parse the PFS section of an Aspen input file to extract block and stream data
    with coordinates and routing information.
    
    Args:
        text: The complete Aspen input file text
        
    Returns:
        Tuple containing dictionaries of Block and Stream objects with visual information
    """
    # Find PFS section
    pfs_section = re.search(r';PFS.*', text, re.DOTALL)
    if not pfs_section:
        raise ValueError("PFS section not found in the input file")
    
    pfs_text = pfs_section.group(0)
    
    # Extract blocks with coordinates
    blocks = {}
    for block_match in BLOCK_SECTION_RE.finditer(pfs_text):
        block_id = block_match.group(1)
        block = Block(block_id)
        
        # Find position in text right after this block ID
        start_pos = block_match.end()
        end_pos = pfs_text.find(";BLOCK", start_pos)
        if end_pos == -1:
            end_pos = pfs_text.find(";STREAM", start_pos)
        
        block_text = pfs_text[start_pos:end_pos]
        
        # Extract coordinates
        coord_match = AT_COORD_RE.search(block_text)
        if coord_match:
            x, y = float(coord_match.group(1)), float(coord_match.group(2))
            block.x_coord = x
            block.y_coord = y
        
        blocks[block_id] = block
    
    # Extract streams with routes
    streams = {}
    for stream_match in STREAM_SECTION_RE.finditer(pfs_text):
        stream_id = stream_match.group(1)
        stream = Stream(ID=stream_id)
        
        # Find position in text right after this stream ID
        start_pos = stream_match.end()
        end_pos = pfs_text.find(";STREAM", start_pos)
        if end_pos == -1:
            end_pos = len(pfs_text)
        
        stream_text = pfs_text[start_pos:end_pos]
        
        # Extract stream coordinates
        coord_match = AT_COORD_RE.search(stream_text)
        if coord_match:
            x, y = float(coord_match.group(1)), float(coord_match.group(2))
            stream.x_coord = x
            stream.y_coord = y
        
        # Extract routes
        stream.routes = []
        current_route = None
        
        for line in stream_text.split('\n'):
            route_start = ROUTE_START_RE.search(line)
            if route_start:
                route_num = int(route_start.group(1))
                # Start a new route
                current_route = Route(route_num)
                stream.routes.append(current_route)
                continue
            
            route_point = ROUTE_POINT_RE.search(line)
            if current_route is not None and route_point:
                point_type = route_point.group(1)  # r, t, x, y
                direction = route_point.group(2)   # r, l, u, d, x, y
                x = float(route_point.group(3))
                y = float(route_point.group(4))
                z = int(route_point.group(5))
                
                point = RoutePoint(point_type, direction, x, y, z)
                current_route.points.append(point)
        
        streams[stream_id] = stream
    
    return blocks, streams

def extract_flowsheet_connections(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the FLOWSHEET section to extract block connections.
    
    Args:
        text: The complete Aspen input file text
        
    Returns:
        Dictionary mapping block IDs to lists of (input_stream, output_stream) tuples
    """
    # Find FLOWSHEET section
    flowsheet_section = re.search(r'FLOWSHEET\s*(.*?)(?:\n\s*\n|\n[A-Z])', text, re.DOTALL)
    if not flowsheet_section:
        raise ValueError("FLOWSHEET section not found in the input file")
    
    flowsheet_text = flowsheet_section.group(1)
    
    # Parse block connections
    connections = {}
    block_re = re.compile(r'BLOCK\s+([A-Za-z0-9_]+)\s+IN=([A-Za-z0-9_ ]+)\s+OUT=([A-Za-z0-9_ ]+)')
    
    for line in flowsheet_text.split('\n'):
        match = block_re.search(line)
        if match:
            block_id = match.group(1)
            inputs = match.group(2).strip().split()
            outputs = match.group(3).strip().split()
            connections[block_id] = (inputs, outputs)
    
    return connections