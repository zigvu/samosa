#!/usr/bin/env python

import logging
import argparse

import _init_paths
from tools.frames.frame_extractor import FrameExtractor

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--video_path', dest='video_path', help='Path of video file to use',
                        required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fps', dest='fps', help='FPS to extract frames at',
                        type=int)
    group.add_argument('--frame_numbers', dest='frame_numbers_file',
                        help='File with frame numbers')
    parser.add_argument('--output_path', dest='output_path', help='Output folder path',
                        required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    logging.debug('Start extracting frames from {}'.format(args.video_path))

    frameExtractor = FrameExtractor(args.video_path, args.output_path)
    if args.fps != None:
        fnFileMap = frameExtractor.extract_sequential(args.fps)
    elif args.frame_numbers_file != None:
        fnFileMap = frameExtractor.extract_based_on_file(args.frame_numbers_file)
    else:
        raise RuntimeError("Please specify either `fps` or `frame_numbers`")
    logging.debug('Extracted {} frames to {}'.format(len(fnFileMap.keys()), args.output_path))
