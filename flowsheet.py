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

file_path = ""

@dataclass
class Block:
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: float = 0
    y_coord: float = 0

@dataclass
class Stream:
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: float = 0
    y_coord: float = 0
    # route 0 0
    x_coord_route: List[List[float]] = 0
    y_coord_route: List[List[float]] = 0


def parse_flowsheet(text: str) -> Dict[str, Block]:
    """
    Parse BLOCK lines and return {block_ID: Block(ID, inputs, outputs)}.
    """
    blocks: Dict[str, Block] = {}
    streams: Dict[str, Stream] = {}

    for m in LINE_RE.finditer(text):
        block_id, rest = m.group(1), m.group(2)
        blk = blocks.get(block_id) or Block(block_id)

        mi = IN_RE.search(rest)
        mo = OUT_RE.search(rest)

        if mi:
            blk.inputs = mi.group(1).split()
        if mo:
            blk.outputs = mo.group(1).split()

        blocks[block_id] = blk


        for s in blk.inputs:
                st = streams.get(s)
                if st is None:
                    st = Stream(ID=s)
                    streams[s] = st
                if block_id not in st.outputs:
                    st.outputs.append(block_id)

            # Streams flowing OUT OF this block → stream.inputs include this block
        for s in blk.outputs:
            st = streams.get(s)
            if st is None:
                st = Stream(ID=s)
                streams[s] = st
            if block_id not in st.inputs:
                st.inputs.append(block_id)

    return blocks, streams

def reading_in_flowsheet():
    # Tk().withdraw()
    # global file_path
    # # Prompt user to select a file with .inp extension
    # file_path = filedialog.askopenfilename(
    #     title="Select a .inp file",
    #     filetypes=[("Input files", "*.inp")]
    # )
    #testing
    global file_path
    # file_path = r"C:\Users\Wiltj\LU Student Dropbox\Wil Jaques\Year 4 Lehigh\CSE 281\Air-Products-Capstone\co2 conditioning (visual).inp"
    file_path = r"C:\Users\Wiltj\LU Student Dropbox\Wil Jaques\Year 4 Lehigh\CSE 281\Air-Products-Capstone\netlc3rg (visual).inp"
    if not file_path:
        raise FileNotFoundError("No file selected.")


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


def entering_blocks(blocks: Dict[str, Block]) -> str:
    with open(file_path, 'r') as file:
        file_contents = file.read()
        lines = file_contents.splitlines()

    idx = 0
    while idx < len(lines):
        if lines[idx] == ";BLOCK":
            current_block_id = None
        elif ";ID:" in lines[idx]:
            current_block_id = lines[idx].split(":")[1].strip()
            if current_block_id in blocks:
                # Version
                idx += 1
                # Icon
                idx += 1
                # Flag
                idx += 1
                # Section
                idx += 1
                # Move to ;At
                idx += 1
                if ";At" in lines[idx]:
                    b = blocks[current_block_id]
                    print(f"Setting block {b.ID} at ({b.x_coord:.1f}, {b.y_coord:.1f})")
                    lines[idx] = f";At {b.x_coord:.1f} {b.y_coord:.1f}"
                    # Label At
                    idx += 1
                    # Scale
                    idx += 1
                else:
                    raise ValueError(f";At line not found for block {current_block_id}")
        idx += 1

    with open(file_path, 'w') as f:
        f.writelines("\n".join(lines))
    return "\n".join(lines)



def assign_coordinates(blocks: Dict[str, Block],
                       layers: List[List[str]],
                       dx: float = 10.0,
                       dy: float = 6.0,
                       spread: float = 1.0) -> None:
    """
    Assign x,y coordinates directly to each Block in `blocks`.
    - dx, dy: base spacing
    - spread: global multiplier to spread everything out uniformly
    """
    for lx, layer in enumerate(layers):
        n = max(1, len(layer))
        # auto vertical spacing so tall layers don't get cramped
        layer_dy = dy * spread * max(1.0, min(1.0 + 0.12*(n-1), 2.0))
        y0 = -(n-1)/2.0 * layer_dy
        x = (dx * spread) * lx
        for i, bid in enumerate(layer):
            y = y0 + i * layer_dy
            blk = blocks[bid]
            blk.x_coord = float(x)
            blk.y_coord = float(y)

def rescale_coords(coords, factor=1.25):
    """Uniformly scale all coordinates (quick 'zoom out')."""
    return {k: (x*factor, y*factor) for k,(x,y) in coords.items()}

def draw_flowsheet(blocks: Dict[str, Block], figsize=(14, 8), margin: float = 6.0):
    input_to_blocks, _ = build_index(blocks)

    xs = [b.x_coord for b in blocks.values()]
    ys = [b.y_coord for b in blocks.values()]
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for b in blocks.values():
        x1, y1 = b.x_coord, b.y_coord
        for o in b.outputs:
            for nb in input_to_blocks.get(o, []):
                x2, y2 = blocks[nb].x_coord, blocks[nb].y_coord
                ax.plot([x1, x2], [y1, y2])

    # nodes + labels
    for b in blocks.values():
        ax.scatter([b.x_coord], [b.y_coord], s=160)
        ax.text(b.x_coord, b.y_coord + 0.7, b.ID, ha='center', va='bottom', fontsize=9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.set_title("Auto-arranged flowsheet")
    ax.set_xlabel("Layer (left → right)")
    ax.set_ylabel("Relative position")
    plt.show()



def find_block_for_stream(blocks: dict, stream: str):
    """
    Return the ID of the first block where `stream`
    appears in either inputs or outputs. None if not found.
    """
    for key, blk in blocks.items():
        if stream in blk.inputs or stream in blk.outputs:
            return key
    return None


def entering_streams(blocks: Dict[str, Block],streams: Dict[str, Stream]) -> str:
    with open(file_path, 'r') as f:
        file_contents = f.read()

    lines = file_contents.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip() == ";STREAM":
            # Expect an ;ID: line next (but be defensive)
            j = i + 1
            if j >= n or not lines[j].lstrip().startswith(";ID:"):
                i += 1
                continue

            # Parse stream ID
            current_stream_id = lines[j].split(":", 1)[1].strip()

            # Find the ;At line within this STREAM block
            k = j + 1
            while k < n and not lines[k].lstrip().startswith(";At"):
                k += 1

            if k < n and current_stream_id in streams:
                s = streams[current_stream_id]

                # Prefer producing block(s) for placement
                candidate_block_id = None
                # Choose the producer with the largest x (rightmost), if multiple
                producers = [bid for bid in s.inputs if bid in blocks]
                if producers:
                    candidate_block_id = max(producers, key=lambda b: blocks[b].x_coord)
                else:
                    # Fallback: choose a consumer (e.g., leftmost or first)
                    consumers = [bid for bid in s.outputs if bid in blocks]
                    if consumers:
                        candidate_block_id = min(consumers, key=lambda b: blocks[b].x_coord)

                if candidate_block_id is not None:
                    b = blocks[candidate_block_id]
                    x, y = b.x_coord + 10, b.y_coord
                    lines[k] = f";At {x:.1f} {y:.1f}"
                    # Optional: print for debugging
                    # print(f"Placed stream {current_stream_id} near {candidate_block_id} at ({x:.1f}, {y:.1f})")

            # Advance past this STREAM block’s ;At line if we found it
            i = k
        i += 1

    out = "\n".join(lines)
    with open(file_path, 'w') as f:
        f.write(out)
    return out

# ---------- example usage ----------
if __name__ == "__main__":
    flowsheet_text = reading_in_flowsheet()

    blocks, streams = parse_flowsheet(flowsheet_text)
    roots = find_first_elements(blocks)
    layers = build_layers(blocks, roots)

    assign_coordinates(blocks, layers, dx=3, dy=5)   # writes into each Block
    # rescale_block_coords(blocks, factor=1.2)       # optional

    draw_flowsheet(blocks, figsize=(16, 9), margin=8.0)

    entering_blocks(blocks)
    # entering_streams(blocks,streams)
