"""
Main module for Aspen flowsheet visualization optimization.

This module integrates parsing, optimization, routing, visualization, and
writing functions to optimize an Aspen flowsheet layout.
"""

import os
import argparse
import matplotlib.pyplot as plt
from parsing import parse_flowsheet_pfs, extract_flowsheet_connections
from optimizer import optimize_block_positions
from routing import calculate_stream_paths
from visualization import draw_optimized_flowsheet
from aspen_writer import update_aspen_file

def main():
    """
    Main function to run the flowsheet optimization process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize Aspen flowsheet layout")
    parser.add_argument("input_file", help="Path to input Aspen file")
    parser.add_argument("--output", "-o", help="Path for output Aspen file")
    parser.add_argument("--visualize", "-v", action="store_true", help="Show visualization")
    parser.add_argument("--save-image", "-s", help="Path to save visualization image")
    parser.add_argument("--iterations", "-i", type=int, default=50, help="Number of optimization iterations")
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = f"{base}_optimized{ext}"
    
    # Set default image path if not specified but save requested
    if args.save_image is None and args.visualize:
        base, _ = os.path.splitext(args.input_file)
        args.save_image = f"{base}_visualization.png"
    
    # Read and parse input file
    print(f"Reading input file: {args.input_file}")
    with open(args.input_file, 'r') as f:
        content = f.read()
    
    # Extract block and stream information
    print("Extracting flowsheet elements...")
    blocks, streams = parse_flowsheet_pfs(content)
    
    # Extract connections from FLOWSHEET section
    connections = extract_flowsheet_connections(content)
    
    # Update blocks with connection information
    for block_id, (inputs, outputs) in connections.items():
        if block_id in blocks:
            blocks[block_id].inputs = inputs
            blocks[block_id].outputs = outputs
    
    # Connect streams to blocks
    for stream_id, stream in streams.items():
        for block_id, (inputs, outputs) in connections.items():
            if stream_id in inputs:
                stream.to_block = block_id
            if stream_id in outputs:
                stream.from_block = block_id
    
    # Optimize block positions
    print(f"Optimizing block positions ({args.iterations} iterations)...")
    optimized_blocks = optimize_block_positions(
        blocks, 
        streams,
        iterations=args.iterations,
        preserve_layers=True
    )
    
    # Calculate optimal stream paths
    print("Calculating optimal stream paths...")
    stream_paths = calculate_stream_paths(optimized_blocks, streams)
    
    # Visualize if requested
    if args.visualize or args.save_image:
        print("Creating visualization...")
        fig = draw_optimized_flowsheet(
            optimized_blocks,
            streams,
            stream_paths,
            output_path=args.save_image
        )
        
        if args.visualize:
            plt.show()
    
    # Write updated Aspen file
    print(f"Writing optimized Aspen file: {args.output}")
    update_aspen_file(
        args.input_file,
        args.output,
        optimized_blocks,
        stream_paths
    )
    
    print("Done!")

if __name__ == "__main__":
    main()