import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt



# Tokens are alnum + underscore
TOKEN = r"[A-Za-z0-9_]+"

# Match a flowsheet line defining a block
LINE_RE = re.compile(rf"^\s*BLOCK\s+({TOKEN})\s+(.*?)\s*$", re.MULTILINE)

# Extract IN=... stopping before OUT= or end of line
IN_RE  = re.compile(rf"\bIN=([A-Za-z0-9_ ]+?)(?=\s+OUT=|\s*$)")
# Extract OUT=... to end of line
OUT_RE = re.compile(rf"\bOUT=([A-Za-z0-9_ ]+)\s*$")

@dataclass
class Block:
    ID: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    x_coord: int = 0
    y_coord: int = 0

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

    return blocks

# ---------- example usage ----------
if __name__ == "__main__":
    flowsheet_text = """
    FLOWSHEET
        BLOCK COOL1 IN=CO2FEED CW1 OUT=TOSEP1 HW1
        BLOCK SEP1 IN=TOSEP1 OUT=V1 L1
        BLOCK COMP1 IN=V1 V5 OUT=TOHX1
        BLOCK COOL2 IN=TOCOL2 CW2 OUT=TOSEP2 HW2
        BLOCK SEP2 IN=TOSEP2 OUT=V2 L2
        BLOCK COMP2 IN=V2 OUT=TOCOL3
        BLOCK COOL3 IN=TOCOL3 CW3 OUT=TOSEP3 HW3
        BLOCK SEP3 IN=TOSEP3 OUT=V3 L3
        BLOCK COMP3 IN=V3 OUT=TOCOL4
        BLOCK SEP4 IN=TOSEP4 OUT=WETGAS L4
        BLOCK COMP4 IN=DRYGAS OUT=TOCOL5
        BLOCK COOL5 IN=TOCOL5 CW5 OUT=TOPUMP HW5
        BLOCK PUMP1 IN=TOPUMP OUT=TOCOL6
        BLOCK ABS IN=TOABS WETGAS OUT=DRYGAS TOVLV
        BLOCK REG IN=TOREG OUT=GAS REGBOT
        BLOCK MU IN=TOMU TEGMU OUT=TOPUMP2
        BLOCK PUMP2 IN=TOPUMP2 OUT=TOABS
        BLOCK GCO IN=O2 TOGCO OUT=OXDPROD
        BLOCK HX1 IN=OXDPROD TOHX1 OUT=TOCOL2 TOSCO
        BLOCK COOL7 IN=GAS CW7 OUT=CO2RCY HW7
        BLOCK SEP5 IN=CO2RCY OUT=V5 L5
        BLOCK COOL4 IN=TOCOL4 CW4 OUT=TOSEP4 HW4
        BLOCK COOL6 IN=TOCOL6 CW6 OUT=TOTRAN HW6
        BLOCK MIX IN=HW6 HW5 HW3 HW4 HW2 HW1 HW7 OUT=TOTW
        BLOCK VLV IN=TOVLV OUT=TOHX2
        BLOCK SCO IN=TOSCO FEO OUT=TOSPLIT
        BLOCK HX2 IN=REGBOT TOHX2 OUT=TOCOL8 TOREG
        BLOCK COOL8 IN=TOCOL8 CW8 OUT=TOMU HW8
        BLOCK SPLIT IN=TOSPLIT OUT=TOGCO FES
    """

    blocks = parse_flowsheet(flowsheet_text)
    for ID, b in blocks.items():
        print(f"{ID}: IN={b.inputs}  OUT={b.outputs}")



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

print("last element: ", find_last_elements(blocks))


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

def draw_flowsheet(blocks: Dict[str, Block], coords: Dict[str, Tuple[float,float]]):
    input_to_blocks, _ = build_index(blocks)
    fig, ax = plt.subplots(figsize=(12, 6))
    for b in blocks.values():
        x1, y1 = coords[b.ID]
        for o in b.outputs:
            for nb in input_to_blocks.get(o, []):
                x2, y2 = coords[nb]
                ax.plot([x1, x2], [y1, y2])
    for bid, (x,y) in coords.items():
        ax.scatter([x], [y], s=150)
        ax.text(x, y+0.6, bid, ha='center', va='bottom', fontsize=9)
    ax.set_xlabel("Layer (left → right)")
    ax.set_ylabel("Relative position")
    ax.set_title("Auto-arranged flowsheet (schematic)")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    plt.show()

def emit_pfsdata(blocks: Dict[str, Block], coords: Dict[str, Tuple[float,float]]) -> str:
    lines = []
    lines.append(";PFSVData\n")
    lines.append(f";# of PFS Objects = {len(blocks)}\n")
    lines.append(";SIZE 1.000000 0.000000 0.000000 100.000000\n")
    for bid, (x,y) in coords.items():
        lines.extend([
            ";BLOCK\n",
            f";ID: {bid}\n",
            ";Version: 1\n",
            ';ICON: "BLOCK"\n',
            ";Flag 0\n",
            ";Section GLOBAL\n",
            f";At {x:.6f} {y:.6f}\n",
            ";Label At 0.000000 0.750000\n",
            ";Scale 1.000000 Modifier 0\n"
        ])
    return "".join(lines)

blocks = parse_flowsheet(flowsheet_text)
roots = find_first_elements(blocks)
end = find_last_elements(blocks)
layers = build_layers(blocks, roots)
coords = assign_coordinates(layers, dx=8.0, dy=4.0)

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

def draw_flowsheet(blocks, coords, figsize=(14, 8), margin=6.0):
    input_to_blocks, _ = build_index(blocks)

    # compute bounds for nicer framing
    xs, ys = zip(*coords.values())
    xmin, xmax = min(xs)-margin, max(xs)+margin
    ymin, ymax = min(ys)-margin, max(ys)+margin

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for b in blocks.values():
        x1, y1 = coords[b.ID]
        for o in b.outputs:
            for nb in input_to_blocks.get(o, []):
                x2, y2 = coords[nb]
                ax.plot([x1, x2], [y1, y2])  # default style

    # nodes + labels
    for bid, (x,y) in coords.items():
        ax.scatter([x], [y], s=160)
        ax.text(x, y+0.7, bid, ha='center', va='bottom', fontsize=9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.set_title("Auto-arranged flowsheet")
    ax.set_xlabel("Layer (left → right)")
    ax.set_ylabel("Relative position")
    plt.show()

layers = build_layers(blocks, roots)

# Option A: just increase spacing
coords = assign_coordinates(layers, dx=1, dy=1000.0, spread=1)

# Option B: scale after the fact
coords = assign_coordinates(layers, dx=10.0, dy=50.0)
coords = rescale_coords(coords, factor=1.5)

draw_flowsheet(blocks, coords, figsize=(16,9), margin=8.0)
