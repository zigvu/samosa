#!/usr/bin/env python

import logging
import os
import subprocess
import glob
import argparse
import json
import _init_paths

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.pipeline.run_pipeline import RunPipeline
from khajuri.multi.clip import Clip

from chia.configs.chia_config import chia_cfg
from chia.datasets.test_creator import TestCreator
from chia.test.test_config import TestConfig
from chia.test.test_templater import TestTemplater
from chia.test.frame_evaluater import FrameEvaluater

from tools.files.file_utils import FileUtils

from fast_rcnn.config import cfg
from utils.timer import Timer

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test on zigvu model on clips')
    parser.add_argument('--config_file', dest='config_file',
                        help='Path to config file for testing', required=True)
    parser.add_argument('--test_model', dest='test_model',
                        help='Caffe model for testing', required=True)
    parser.add_argument('--clip_folder', dest='clip_folder',
                        help='Path to clips', required=True)
    parser.add_argument('--output_path', dest='output_path',
                        help='Output folder path', required=True)
    parser.add_argument('--no_use_rabbit', dest='no_use_rabbit',
                        help='Use rabbit? (default: True)', action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('pika').setLevel(logging.INFO)
    logging.debug('Start testing.')

    args = parse_args()

    with open(args.config_file) as f:
        configHash = json.load(f)

    config = TestConfig(configHash)
    # config.reset_folders()

    khajuri_cfg.PIPELINE.GPU_IDS = []
    for line in subprocess.check_output(['nvidia-smi', '--list-gpus']).split('\n'):
        if 'GPU ' in line:
            khajuri_cfg.PIPELINE.GPU_IDS += [int(line.split('GPU ')[1].split(':')[0])]

    if args.no_use_rabbit:
        khajuri_cfg.RABBIT.IS_RABBIT_RUN = False

    templater = TestTemplater()

    FileUtils.symlink(args.test_model, chia_cfg.TEST.FILES.CURRENT_MODEL)
    khajuri_cfg.PIPELINE.OUTPUT_BASE_FOLDER = args.output_path

    # keep track of time
    caffeTime = Timer()

    caffeTime.tic()
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
    caffeTime.toc()
    logging.info("Total time for all clip evaluation: {}".format(caffeTime.total_time))
    if len(allClipFiles) > 0:
        logging.info("Avg. time for each clip evaluation: {}".format(
            caffeTime.total_time/len(allClipFiles)
        ))
