#!/usr/bin/env python

import logging
import os
import glob
import argparse

import _init_paths

from khajuri.pipeline.run_pipeline import RunPipeline
from khajuri.multi.clip import Clip

from tools.files.file_utils import FileUtils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test on zigvu model on clips')
    parser.add_argument('--clip_folder', dest='clip_folder',
                        help='Path to clips', required=True)
    parser.add_argument('--output_path', dest='output_path',
                        help='Output folder path', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start testing.')

    args = parse_args()

    runPipeline = RunPipeline()
    runPipeline.start()

    allClipFiles = glob.glob("{}/*.mp4".format(args.clip_folder))
    for clipFile in allClipFiles:
        logging.info("Working on file {}".format(clipFile))
        clipNumber = os.path.splitext(os.path.basename(clipFile))[0]
        clipOutPath = os.path.join(args.output_path, clipNumber)

        clip = Clip()
        clip.clip_id = clipNumber
        clip.clip_path = clipFile
        clip.result_path = os.path.join(clipOutPath, 'clip.pkl')

        runPipeline.clipdbQueue.put(clip)
        logging.debug('Putting clip id {} in queue'.format(clip.clip_id))

    runPipeline.join()
