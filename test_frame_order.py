#!/usr/bin/env python3
"""
Test script to verify that frame ordering is working correctly.
This simulates the same logic used in the Rust application.
"""

import os
import re
from pathlib import Path

def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_123.jpg'"""
    if 'frame_' in filename:
        start = filename.find('frame_') + 6  # Skip "frame_"
        if '.' in filename[start:]:
            end = filename.find('.', start)
            number_str = filename[start:end]
            try:
                return int(number_str)
            except ValueError:
                return 0
    return 0

def test_frame_ordering():
    """Test frame ordering logic"""
    output_dir = Path("/home/chavindu/Desktop/Pipeline_repo/output_frames")
    
    if not output_dir.exists():
        print("Output directory not found!")
        return
    
    # Get all frame files
    frame_files = []
    for file in output_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            frame_files.append(file.name)
    
    if not frame_files:
        print("No frame files found!")
        return
    
    print(f"Found {len(frame_files)} frame files")
    
    # Test alphabetical sorting (old way)
    alphabetical_sorted = sorted(frame_files)
    print("\nFirst 10 files (alphabetical sorting):")
    for file in alphabetical_sorted[:10]:
        print(f"  {file}")
    
    # Test numerical sorting (new way)
    numerical_sorted = sorted(frame_files, key=extract_frame_number)
    print("\nFirst 10 files (numerical sorting):")
    for file in numerical_sorted[:10]:
        frame_num = extract_frame_number(file)
        print(f"  {file} (frame #{frame_num})")
    
    # Check if ordering is different
    if alphabetical_sorted != numerical_sorted:
        print("\n✅ Frame ordering is different - numerical sorting is working!")
    else:
        print("\n⚠️  Frame ordering is the same - might need investigation")
    
    print(f"\nFirst frame (numerical): {numerical_sorted[0]}")
    print(f"Last frame (numerical): {numerical_sorted[-1]}")

if __name__ == "__main__":
    test_frame_ordering()
