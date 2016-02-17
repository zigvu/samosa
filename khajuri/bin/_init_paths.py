"""Set up paths for Fast R-CNN."""

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add samosa to PYTHONPATH
samosa_path = '/home/ubuntu/samosa'
add_path(samosa_path)
