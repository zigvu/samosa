#!/usr/bin/env python

import logging
import os
import glob
import argparse
import json
import pickle

import _init_paths
from chia.configs.chia_config import chia_cfg
from chia.datasets.test_creator import TestCreator
from chia.test.test_config import TestConfig
from chia.test.test_templater import TestTemplater
from chia.test.frame_evaluater import FrameEvaluater

from khajuri.multi.clip import Clip

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

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start testing.')

    args = parse_args()

    with open(args.config_file) as f:
        configHash = json.load(f)

    config = TestConfig(configHash)
    # config.reset_folders()
    templater = TestTemplater()

    FileUtils.symlink(args.test_model, chia_cfg.TEST.FILES.CURRENT_MODEL)

    # keep track of times
    frameExtractorTime = Timer()
    caffeTime = Timer()
    pickleTime = Timer()

    frameEvaluater = FrameEvaluater(cfg.GPU_ID)

    for clipFile in glob.glob("{}/*.mp4".format(args.clip_folder)):
        logging.info("Working on file {}".format(clipFile))
        clipNumber = os.path.splitext(os.path.basename(clipFile))[0]
        clipOutPath = os.path.join(args.output_path, clipNumber)

        clip = Clip()
        clip.clip_id = clipNumber
        clip.clip_path = clipFile
        clip.result_path = os.path.join(clipOutPath, 'clip.pkl')

        # extract frames
        frameExtractorTime.tic()
        framedbPath = os.path.join(clipOutPath, 'framedb')
        testCreator = TestCreator(5, framedbPath)
        clip.framedb = testCreator.get_framedb(clip.clip_path)
        frameExtractorTime.toc()
        # evaluate frames
        caffeTime.tic()
        clip.preddb = frameEvaluater.evaluate(clip.framedb)
        caffeTime.toc()
        # save to pickle
        pickleTime.tic()
        with open(clip.result_path, 'wb') as f:
            clip.framedb = None
            pickle.dump(clip, f)
        pickleTime.toc()
        logging.info("Avg. times: FrameExtractor {}, Caffe: {}, Pickle: {}".format(
            frameExtractorTime.average_time, caffeTime.average_time, pickleTime.average_time))

    logging.info("Total times: FrameExtractor {}, Caffe: {}, Pickle: {}".format(
        frameExtractorTime.total_time, caffeTime.total_time, pickleTime.total_time))
    # from IPython import embed; embed()
