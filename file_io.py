"""
File I/O module for flowsheet processing.

This module contains functions for reading flowsheet files and updating
coordinate information in the input files.
"""

from tkinter import Tk, filedialog
from typing import Tuple, Dict

def reading_in_flowsheet() -> Tuple[str, str]:
    """
    Prompt user to select a .inp file and extract flowsheet data.
    
    Returns:
        Tuple containing:
        - flowsheet_text: String containing extracted BLOCK lines
        - file_path: Path to the selected file
        
    Raises:
        FileNotFoundError: If no file is selected
        ValueError: If no 'FLOWSHEET' section is found in the file
    """
    Tk().withdraw()
    # Prompt user to select a file with .inp extension
    file_path = filedialog.askopenfilename(
        title="Select a .inp file",
        filetypes=[("Input files", "*.inp")]
    )

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

    return flowsheet_text, file_path

def blocks(file_path: str, coords: Dict[str, Tuple[float, float]]) -> str:
    """
    Parse the input file to find blocks and update their coordinates.
    
    Args:
        file_path: Path to the input file
        coords: Dictionary mapping block IDs to (x, y) coordinate tuples
        
    Returns:
        Updated file contents as a string
        
    Raises:
        ValueError: If ;At line is not found for a block
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
                    # Version
                    idx += 1
                    version_line = lines[idx]
                    # Icon Line
                    idx += 1
                    icon_line = lines[idx]
                    # Flag line
                    idx += 1
                    flag_line = lines[idx]
                    # Section line
                    idx += 1
                    section_line = lines[idx]

                    # moving to next line to find ;AT
                    idx += 1
                    if ";At" in lines[idx]:
                        x, y = coords[current_block_id]
                        print("Lines ", lines[idx])
                        # Replace the coordinates in the ";AT" line
                        print(f"Setting block {current_block_id} to coordinates ({x:.1f}, {y:.1f})")
                        lines[idx] = f";At {x:.1f} {y:.1f}"

                        # Label at line
                        idx += 1
                        label_at_line = lines[idx]
                        # scale line
                        idx += 1
                        scale_line = lines[idx]

                    else:
                        raise ValueError(f";At line not found for block {current_block_id}")
            idx += 1

    updated_content = "\n".join(lines)
    
    # Save the updated file contents back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)

    return updated_content

def streams(file_path: str, coords: Dict[str, Tuple[float, float]]) -> str:
    """
    Parse the input file to find streams and update their coordinates.
    
    Args:
        file_path: Path to the input file
        coords: Dictionary mapping stream IDs to (x, y) coordinate tuples
        
    Returns:
        Updated file contents as a string
        
    Raises:
        ValueError: If ;At line is not found for a stream
    """
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
                    # Version
                    idx += 1
                    version_line = lines[idx]
                    # Flag line
                    idx += 1
                    flag_line = lines[idx]
                    # Section line
                    idx += 1
                    section_line = lines[idx]
                    # Type Line
                    idx += 1
                    icon_line = lines[idx]

                    # moving to next line to find ;AT
                    idx += 1
                    if ";At" in lines[idx]:
                        x, y = coords[current_stream_id]
                        print("Lines ", lines[idx])
                        # Replace the coordinates in the ";AT" line
                        print(f"Setting stream {current_stream_id} to coordinates ({x:.1f}, {y:.1f})")
                        lines[idx] = f";At {x:.1f} {y:.1f}"

                        # Label at line
                        idx += 1
                        label_at_line = lines[idx]
                        # scale line
                        idx += 1
                        scale_line = lines[idx]

                    else:
                        raise ValueError(f";At line not found for stream {current_stream_id}")
            idx += 1

    updated_content = "\n".join(lines)
        
    # Save the updated file contents back to the file
    with open(file_path, 'w') as f:
            f.write(updated_content)

    return updated_content