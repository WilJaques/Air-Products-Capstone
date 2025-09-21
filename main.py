"""
Main entry point for the flowsheet processing application.

This module ties together all the other modules and provides the main
function to run the flowsheet generation process.
"""

import matplotlib.pyplot as plt
from file_io import reading_in_flowsheet
from parsing import parse_flowsheet, parse_streams
from layout import find_first_elements, build_layers, assign_coordinates
from visualization import draw_flowsheet, stream_coords

def main():
    """
    Main function to run the flowsheet generation process.
    
    This function orchestrates the entire workflow:
    1. Read and parse the flowsheet file
    2. Determine layout and assign coordinates
    3. Calculate stream paths
    4. Display the flowsheet visualization
    """
    try:
        flowsheet_text, file_path = reading_in_flowsheet()
        blocks_data = parse_flowsheet(flowsheet_text)
        
        if not blocks_data:
            print("No blocks were parsed. Please check the input file format.")
            return

        roots = find_first_elements(blocks_data)
        if not roots:
            print("Warning: Could not determine starting blocks for the layout.")
            # Fallback: use all blocks with no inputs, or the first block if none exist
            roots = [bid for bid, b in blocks_data.items() if not b.inputs]
            if not roots and blocks_data:
                roots = [next(iter(blocks_data))]
        
        layers = build_layers(blocks_data, roots)
        coords = assign_coordinates(layers, dx=12, dy=8, spread=1.5)

        streams_data = parse_streams(flowsheet_text, blocks_data)
        stream_pos = stream_coords(streams_data, coords)

        draw_flowsheet(blocks_data, coords, streams=streams_data, 
                      stream_coords_dict=stream_pos, figsize=(16, 10), margin=10.0)

    except FileNotFoundError as e:
        print(f"Operation cancelled: {e}")
    except ValueError as e:
        print(f"Error processing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# ---------- script execution ----------
if __name__ == "__main__":
    main()
    plt.close('all')