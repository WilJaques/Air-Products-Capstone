"""
Data models for flowsheet processing.

This module contains the dataclasses used to represent blocks and streams
in the flowsheet processing system.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Block:
    """
    Represents a block in the flowsheet.
    
    Attributes:
        ID: Unique identifier for the block
        inputs: List of input stream IDs
        outputs: List of output stream IDs
        x_coord: X coordinate for visualization (default: 0)
        y_coord: Y coordinate for visualization (default: 0)
    """
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: int = 0
    y_coord: int = 0

@dataclass
class Stream:
    """
    Represents a stream connection between blocks.
    
    Attributes:
        ID: Unique identifier for the stream
        from_block: ID of the source block (default: empty string)
        to_block: ID of the destination block (default: empty string)
    """
    ID: str
    from_block: str = ""
    to_block: str = ""