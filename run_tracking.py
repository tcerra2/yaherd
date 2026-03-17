#!/usr/bin/env python3
"""Run YOLO tracking with proper module paths"""

import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path so boxmot can be imported
sys.path.insert(0, script_dir)

# Now run the tracking script
if __name__ == "__main__":
    # Import after path is set
    from tracking import track
    
    # Run with command line args
    track.main()
