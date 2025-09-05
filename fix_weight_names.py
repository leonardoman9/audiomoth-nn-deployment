#!/usr/bin/env python3
"""
Fix invalid C variable names in weight files
"""

import re
import sys

def fix_c_names(file_path):
    """Fix invalid characters in C variable names"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace invalid characters with underscores
    # Replace : . / [ ] with _
    content = re.sub(r'[:.\/\[\]]', '_', content)
    
    # Replace multiple underscores with single
    content = re.sub(r'_+', '_', content)
    
    # Remove trailing underscores before array brackets
    content = re.sub(r'_(\[)', r'\1', content)
    
    # Fix any floating point array sizes (like [1.0])
    content = re.sub(r'\[(\d+\.\d+)\]', r'[1]', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed C names in {file_path}")

def main():
    files_to_fix = [
        'backbone_real_weights.h',
        'backbone_real_weights.c', 
        'streaming_real_weights.h',
        'streaming_real_weights.c'
    ]
    
    for file_path in files_to_fix:
        try:
            fix_c_names(file_path)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    print("✅ All weight files fixed!")

if __name__ == "__main__":
    main()