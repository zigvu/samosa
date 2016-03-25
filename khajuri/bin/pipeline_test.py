#!/usr/bin/env python

import logging
import os
import glob
import argparse

import _init_paths

from khajuri.configs.khajuri_config import khajuri_cfg
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

    khajuri_cfg.PIPELINE.IS_DRY_RUN = True
    khajuri_cfg.RABBIT.IS_RABBIT_RUN = False
    runPipeline = RunPipeline()

    allClipFiles = glob.glob("{}/*.mp4".format(args.clip_folder))
    for clipFile in allClipFiles:
        clipNumber = os.path.splitext(os.path.basename(clipFile))[0]

        clip = Clip()
        clip.clip_id = clipNumber
        clip.clip_path = clipFile
        clip.result_path = {
            'base_path': args.output_path,
            'pickle': os.path.join(args.output_path, '{}.pkl'.format(clipNumber)),
            'json': os.path.join(args.output_path, '{}.json'.format(clipNumber))
        }

        runPipeline.clipdbQueue.put(clip)
        logging.debug('RabbitToClip: process clip: {}'.format(clip.clip_id))

    runPipeline.start()
    runPipeline.join()
