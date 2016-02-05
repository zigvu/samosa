#!/usr/bin/env python

import os
import logging
import argparse

import _init_paths
from tools.frames.image_manipulator import ImageManipulator
from khajuri.datasets.annotations.file_reader import FileReader

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Convert annotation to dataset')
    parser.add_argument('--annotation_file', dest='annotation_file', help='Annotation file',
                        required=True)
    parser.add_argument('--frame_file', dest='frame_file', help='Frame file',
                        required=True)
    parser.add_argument('--output_path', dest='output_path', help='Output folder path',
                        required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    fileReader = FileReader(args.annotation_file)
    imageManipulator = ImageManipulator(args.frame_file)
    for cls in fileReader.get_classes():
        for bbox in fileReader.get_bboxes(cls):
            imageManipulator.addLabeledBbox(bbox, cls)
    outputFileName = os.path.join(args.output_path, os.path.basename(args.frame_file))
    imageManipulator.saveImage(outputFileName)

    logging.debug('Saved drawn annotations in {}'.format(outputFileName))
