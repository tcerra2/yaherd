#!/usr/bin/env python3
"""Simple script to run YOLO tracking"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now try to import and run
try:
    from tracking.track import run
    print("✓ BoxMOT module imported successfully!")
    print("\nAvailable tracking methods: bytetrack, botsort, ocsort, deepocsort, strongsort, hybridsort")
    print("\nExample commands:")
    print("  python tracking/track.py --source 0 --yolo-model yolov8n                    # webcam")
    print("  python tracking/track.py --source video.mp4 --yolo-model yolov8n            # video file")
    print("  python tracking/track.py --source 0 --tracking-method botsort               # different tracker")
    print("\nRunning with default settings (webcam, yolov8n, bytetrack)...")
    print("(Press 'q' to quit)\n")
    
except ImportError as e:
    print(f"Error importing: {e}")
    print("\nTrying to diagnose...")
    try:
        import boxmot
        print("✓ boxmot found")
    except:
        print("✗ boxmot not found")
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics found")
    except:
        print("✗ ultralytics not found")
