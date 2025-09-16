import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt



# Tokens are alnum + underscore
TOKEN = r"[A-Za-z0-9_]+"

# Match a flowsheet line defining a block
LINE_RE = re.compile(rf"^\s*BLOCK\s+({TOKEN})\s+(.*?)\s*$", re.MULTILINE)

# Extract IN=... stopping before OUT= or end of line
IN_RE  = re.compile(rf"\bIN=([A-Za-z0-9_ ]+?)(?=\s+OUT=|\s*$)")
# Extract OUT=... to end of line
OUT_RE = re.compile(rf"\bOUT=([A-Za-z0-9_ ]+)\s*$")

# Match a flowsheet line defining a stream
STREAM_LINE_RE = re.compile(rf"^\s*STREAM\s+({TOKEN})\s+(.*?)\s*$", re.MULTILINE)
FROM_RE = re.compile(r"\bFROM=([A-Za-z0-9_]+)")
TO_RE = re.compile(r"\bTO=([A-Za-z0-9_]+)")

file_path = ""

@dataclass
class Block:
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: int = 0
    y_coord: int = 0

@dataclass
class Stream:
    ID: str
    from_block: str = ""
    to_block: str = ""

def parse_flowsheet(text: str) -> Dict[str, Block]:
    """
    Parse BLOCK lines and return {block_ID: Block(ID, inputs, outputs)}.
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

def reading_in_flowsheet():
    Tk().withdraw()
    global file_path
    # Prompt user to select a file with .inp extension
    file_path = filedialog.askopenfilename(
        title="Select a .inp file",
        filetypes=[("Input files", "*.inp")]
    )

    if not file_path:
        raise FileNotFoundError("No file selected.")

    file_path = file_path

    # Read the contents of the selected file
    with open(file_path, 'r') as file:
        file_contents = file.read()

     # Find the start of the flowsheet section
    flowsheet_start = file_contents.find("FLOWSHEET")
    if flowsheet_start == -1:
        raise ValueError("No 'flowsheet' section found in the file.")

    # Extract lines starting with 'BLOCK' after the 'flowsheet' section
    flowsheet_lines = []
    for line in file_contents[flowsheet_start:].splitlines():
        if line.strip().upper().startswith("BLOCK"):
            flowsheet_lines.append(line)

    # Combine the extracted lines into a single string
    flowsheet_text = "\n".join(flowsheet_lines)

    return flowsheet_text


def find_first_elements(blocks: Dict[str, Block]) -> List[str]:
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
    # Collect all outputs from all blocks
    all_inputs = set()
    for block in blocks.values():
        all_inputs.update(block.inputs)

    # Find blocks where none of the inputs are in any outputs
    last_element = []
    for block in blocks.values():
        if block.outputs and all(out not in all_inputs for out in block.outputs):
            last_element.append(block.ID)
    return last_element



def build_flowsheet_tree(blocks: Dict[str, Block], roots: List[str]) -> List[List[str]]:
    """
    Build a tree-like structure where each layer contains block IDs that are connected
    to the previous layer via outputs->inputs.
    Ensures that last elements (from find_last_elements) are placed in the final layer.
    Returns a list of layers, each a list of block IDs.
    """
    # Map input stream -> block(s) that consume it
    input_to_blocks = defaultdict(list)
    for block in blocks.values():
        for inp in block.inputs:
            input_to_blocks[inp].append(block.ID)

    visited = set()
    layers = []
    current_layer = list(roots)
    while current_layer:
        layers.append(current_layer)
        next_layer = []
        for block_id in current_layer:
            if block_id in visited:
                continue
            visited.add(block_id)
            block = blocks[block_id]
            for out in block.outputs:
                for next_block_id in input_to_blocks.get(out, []):
                    if next_block_id not in visited and next_block_id not in next_layer:
                        next_layer.append(next_block_id)
        current_layer = next_layer

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
    return layers


def build_index(blocks: Dict[str, Block]):
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
    Ensures all blocks are included, and last elements (from find_last_elements)
    are in the final layer.
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

def assign_coordinates(layers: List[List[str]], dx: float=10.0, dy: float=6.0) -> Dict[str, Tuple[float,float]]:
    coords = {}
    for lx, layer in enumerate(layers):
        n = len(layer)
        y0 = -(n-1)/2.0 * dy
        for i, bid in enumerate(layer):
            x = lx * dx
            y = y0 + i*dy
            coords[bid] = (x, y)
    return coords

def draw_flowsheet(blocks, coords, streams=None, stream_coords=None, figsize=(14, 8), margin=6.0):
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
    if streams and stream_coords:
        for s in streams.values():
            if s.ID in stream_coords:
                path, label_pos = stream_coords[s.ID]
                
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


def blocks(coords: Dict[str, Tuple[float,float]]) -> str:
    """
    Parse the input file to find blocks and generate pfsdata output.
    """
    # Read the contents of the file
    with open(file_path, 'r') as file:
        file_contents = file.read()

        lines = file_contents.splitlines()
        idx = 0
        while idx < len(lines):
            if lines[idx] == ";BLOCK":
                current_block_id = None  # Reset for a new block
            elif ";ID:" in lines[idx]:
                current_block_id = lines[idx].split(":")[1].strip()
                if current_block_id in coords:
                    # Continue until ";AT" is found
                    #Version
                    idx +=1
                    version_line = lines[idx]
                    # Icon Line
                    idx +=1
                    icon_line = lines[idx]
                    #Flag line
                    idx +=1
                    flag_line = lines[idx]
                    #Section line
                    idx +=1
                    section_line = lines[idx]

                    #moving to next line to find ;AT
                    idx +=1
                    if ";At" in lines[idx]:
                        x, y = coords[current_block_id]
                        print("Lines ", lines[idx])
                        # Replace the coordinates in the ";AT" line
                        print(f"Setting block {current_block_id} to coordinates ({x:.1f}, {y:.1f})")
                        lines[idx] = f";At {x:.1f} {y:.1f}"

                        #Label at line
                        idx +=1
                        label_at_line = lines[idx]
                        #scale line
                        idx +=1
                        scale_line = lines[idx]

                    else:
                        raise ValueError(f";At line not found for block {current_block_id}")
            idx += 1




    #Save the updated file contents back to the file
    with open(file_path, 'w') as f:
        f.writelines("\n".join(lines))

    return file_contents


def assign_coordinates(layers, dx=10.0, dy=6.0, spread=1.0):
    """
    Assign x,y coordinates: x by layer index, y spaced within each layer.
    - dx, dy: base spacing
    - spread: global multiplier to spread everything out uniformly
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

def rescale_coords(coords, factor=1.25):
    """Uniformly scale all coordinates (quick 'zoom out')."""
    return {k: (x*factor, y*factor) for k,(x,y) in coords.items()}

def draw_flowsheet(blocks, coords, streams=None, stream_coords=None, figsize=(14, 8), margin=6.0):
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
    if streams and stream_coords:
        for s in streams.values():
            if s.ID in stream_coords:
                path, label_pos = stream_coords[s.ID]
                
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


def streams():
    with open(file_path, 'r') as file:
        file_contents = file.read()

        lines = file_contents.splitlines()
        idx = 0
        while idx < len(lines):
            if lines[idx] == ";STREAM":
                current_stream_id = None  # Reset for a new stream
            elif ";ID:" in lines[idx]:
                current_stream_id = lines[idx].split(":")[1].strip()
                if current_stream_id in coords:
                    # Continue until ";AT" is found
                    #Version
                    idx +=1
                    version_line = lines[idx]
                    #Flag line
                    idx +=1
                    flag_line = lines[idx]
                    #Section line
                    idx +=1
                    section_line = lines[idx]
                    # Type Line
                    idx +=1
                    icon_line = lines[idx]

                    #moving to next line to find ;AT
                    idx +=1
                    if ";At" in lines[idx]:
                        x, y = coords[current_stream_id]
                        print("Lines ", lines[idx])
                        # Replace the coordinates in the ";AT" line
                        print(f"Setting stream {current_stream_id} to coordinates ({x:.1f}, {y:.1f})")
                        lines[idx] = f";At {x:.1f} {y:.1f}"

                        #Label at line
                        idx +=1
                        label_at_line = lines[idx]
                        #scale line
                        idx +=1
                        scale_line = lines[idx]

                    else:
                        raise ValueError(f";At line not found for block {current_block_id}")
            idx += 1




    #Save the updated file contents back to the file
    with open(file_path, 'w') as f:
        f.writelines("\n".join(lines))

    return file_contents

def stream_coords(streams: Dict[str, Stream], block_coords: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[List[Tuple[float, float]], Tuple[float, float]]]:
    """
    Creates 90-degree angle paths for stream connections with right angle turns.
    Returns: {stream_id: (path_points, label_position)}
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

def parse_streams(flowsheet_text: str, blocks: Dict[str, Block]) -> Dict[str, Stream]:
    """
    Infer streams and their connections from block IN/OUT assignments.
    Returns {stream_ID: Stream(ID, from_block, to_block)}.
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

# ---------- example usage ----------
if __name__ == "__main__":
    flowsheet_text = reading_in_flowsheet()
    blocks = parse_flowsheet(flowsheet_text)
    roots = find_first_elements(blocks)
    layers = build_layers(blocks, roots)
    coords = assign_coordinates(layers, dx=10, dy=20)

    # Parse streams
    streams = parse_streams(flowsheet_text, blocks)
    stream_pos = stream_coords(streams, coords)

    # Draw with streams
    draw_flowsheet(blocks, coords, streams=streams, stream_coords=stream_pos, figsize=(16,9), margin=8.0)
