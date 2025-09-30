"""
Models module for flowsheet processing.

This module defines data structures for blocks, streams, and routes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Block:
    """Represents a process block in a flowsheet."""
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: float = 0.0
    y_coord: float = 0.0
    block_type: str = ""
    icon: str = ""

@dataclass
class RoutePoint:
    """Represents a point in a stream route."""
    point_type: str  # r (root), t (terminal), x (horizontal), y (vertical)
    direction: str   # r (right), l (left), u (up), d (down), x, y
    x: float
    y: float
    z: int

@dataclass
class Route:
    """Represents a route for a stream path."""
    route_id: int
    points: List[RoutePoint] = field(default_factory=list)

@dataclass
class Stream:
    """Represents a material stream in a flowsheet."""
    ID: str
    from_block: str = ""
    to_block: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: float = 0.0
    y_coord: float = 0.0
    routes: List[Route] = field(default_factory=list)