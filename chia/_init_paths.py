"""Set up paths for Fast R-CNN."""

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

CHIA_ROOT = os.path.dirname(__file__)

# py-faster-rcnn path
py_faster_rcnn_path = os.path.join(CHIA_ROOT, '..', 'py-faster-rcnn')

# Add caffe to PYTHONPATH
caffe_path = os.path.join(py_faster_rcnn_path, 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = os.path.join(py_faster_rcnn_path, 'lib')
add_path(lib_path)
