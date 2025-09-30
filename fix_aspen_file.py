"""
Command-line tool to fix formatting issues in Aspen input files.
"""

import re
import sys
import os

def fix_aspen_file(input_file, output_file=None):
    """
    Fix formatting issues in an Aspen input file.
    
    Args:
        input_file: Path to problematic Aspen file
        output_file: Path to save fixed file (defaults to input_fixed.inp)
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
        
    print(f"Reading file: {input_file}")
    with open(input_file, 'r') as f:
        content = f.read()
    
    print("Fixing formatting issues...")
    
    # Fix 1: Ensure dollar signs at beginning of lines don't have empty lines before them
    content = re.sub(r'\n\n(\$ )', r'\n;\1', content)
    
    # Fix 2: Ensure all lines in route sections start with semicolon or dollar sign
    content = re.sub(r'\n([rt$xy][^;\n])', r'\n;\1', content)
    
    # Fix 3: Fix cases where commands are joined on one line
    content = re.sub(r'(\$ \$ 0\.0 0\.0);([A-Z])', r';\$ \$ 0.0 0.0\n;\2', content)
    
    # Fix 4: Ensure consistent formatting for ROUTE declarations
    content = re.sub(r';ROUTE (\d+) (\d+)\n', r';ROUTE \1 \2\n;\n', content)
    
    # Fix 5: Fix incorrect point type/direction combinations
    def fix_point_direction(match):
        pt_type = match.group(1)
        direction = match.group(2)
        
        if pt_type == 'r' or pt_type == 't':
            # For root/terminal points, keep original direction
            return match.group(0)
        else:
            # For x/y points, match the direction to type
            return f";{pt_type} {pt_type}"
    
    content = re.sub(r';([rtxy]) ([rludxy0])', fix_point_direction, content)
    
    # Fix 6: Remove any completely empty lines in route sections
    in_route_section = False
    fixed_lines = []
    
    for line in content.split('\n'):
        # Check if we're entering or exiting a route section
        if line.strip().startswith(';ROUTE'):
            in_route_section = True
            fixed_lines.append(line)
        elif in_route_section and (line.strip() == "" or line.strip() == ";"):
            # Skip completely empty lines or lone semicolons in route sections
            continue
        elif in_route_section and line.strip().startswith(';STREAM'):
            # Exiting route section
            in_route_section = False
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    print(f"Writing fixed file: {output_file}")
    with open(output_file, 'w') as f:
        f.write(content)
    
    print("Done!")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fix_aspen_file(input_file, output_file)
    else:
        print("Usage: python fix_aspen_file.py input.inp [output.inp]")