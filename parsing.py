"""
Parsing module for flowsheet processing.

This module contains functions and regex patterns for parsing flowsheet
text files and extracting block and stream information.
"""

import re
from typing import Dict
from models import Block, Stream

# Tokens are alnum + underscore
TOKEN = r"[A-Za-z0-9_]+"

# Match a flowsheet line defining a block
LINE_RE = re.compile(rf"^\s*BLOCK\s+({TOKEN})\s+(.*?)\s*$", re.MULTILINE)

# Extract IN=... stopping before OUT= or end of line
IN_RE = re.compile(rf"\bIN=([A-Za-z0-9_ ]+?)(?=\s+OUT=|\s*$)")
# Extract OUT=... to end of line
OUT_RE = re.compile(rf"\bOUT=([A-Za-z0-9_ ]+)\s*$")

# Match a flowsheet line defining a stream
STREAM_LINE_RE = re.compile(rf"^\s*STREAM\s+({TOKEN})\s+(.*?)\s*$", re.MULTILINE)
FROM_RE = re.compile(r"\bFROM=([A-Za-z0-9_]+)")
TO_RE = re.compile(r"\bTO=([A-Za-z0-9_]+)")

def parse_flowsheet(text: str) -> Dict[str, Block]:
    """
    Parse BLOCK lines and return {block_ID: Block(ID, inputs, outputs)}.
    
    Args:
        text: The flowsheet text to parse
        
    Returns:
        Dictionary mapping block IDs to Block objects
    """
    blocks: Dict[str, Block] = {}

    for m in LINE_RE.finditer(text):
        ID, rest = m.group(1), m.group(2)
        blk = blocks.get(ID) or Block(ID)
        # Find inputs/outputs independently (order on the line can vary)
        mi = IN_RE.search(rest)
        mo = OUT_RE.search(rest)

        if mi:
            blk.inputs = mi.group(1).split()
        if mo:
            blk.outputs = mo.group(1).split()

        blocks[ID] = blk

    # # Debug: print parsed blocks
    # for ID, b in blocks.items():
    #     print(f"{ID}: IN={b.inputs}  OUT={b.outputs}")

    return blocks

def parse_streams(flowsheet_text: str, blocks: Dict[str, Block]) -> Dict[str, Stream]:
    """
    Infer streams and their connections from block IN/OUT assignments.
    
    Args:
        flowsheet_text: The flowsheet text (not currently used but kept for consistency)
        blocks: Dictionary of parsed blocks
        
    Returns:
        Dictionary mapping stream IDs to Stream objects with from/to block connections
    """
    # Map stream -> from_block and to_block
    stream_to_from_block = {}
    stream_to_to_block = {}

    for block in blocks.values():
        for s in block.outputs:
            stream_to_from_block[s] = block.ID
        for s in block.inputs:
            stream_to_to_block[s] = block.ID

    # All streams mentioned in IN/OUT
    all_streams = set(stream_to_from_block) | set(stream_to_to_block)
    streams = {}
    for s in all_streams:
        streams[s] = Stream(
            ID=s,
            from_block=stream_to_from_block.get(s, ""),
            to_block=stream_to_to_block.get(s, "")
        )
    return streams