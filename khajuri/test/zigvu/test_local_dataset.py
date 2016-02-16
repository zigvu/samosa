#!/usr/bin/env python

import logging
import os
import glob
import argparse
import json
import pickle

import _init_paths
from tools.files.file_utils import FileUtils
from khajuri.datasets.test_creator import TestCreator

from chia.test.test_config import TestConfig
from chia.test.test_templater import TestTemplater
from chia.test.frame_evaluater import FrameEvaluater

from fast_rcnn.config import cfg
from utils.timer import Timer

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test on zigvu model on clip')
    parser.add_argument('--config_file', dest='config_file',
                        help='Path to config file for training', required=True)
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
    logging.debug('Start training.')

    args = parse_args()

    with open(args.config_file) as f:
        configHash = json.load(f)

    config = TestConfig(configHash)
    # config.reset_folders()
    templater = TestTemplater()

    FileUtils.symlink(args.test_model, cfg.ZIGVU.FILES.CURRENT_MODEL)

    # keep track of times
    frameExtractorTime = Timer()
    caffeTime = Timer()
    pickleTime = Timer()

    frameEvaluater = FrameEvaluater(cfg.GPU_ID)

    for clipFile in glob.glob("{}/*.mp4".format(args.clip_folder)):
        logging.info("Working on file {}".format(clipFile))
        clipNumber = os.path.splitext(os.path.basename(clipFile))[0]
        clipOutPath = os.path.join(args.output_path, clipNumber)
        # extract frames
        frameExtractorTime.tic()
        framedbPath = os.path.join(clipOutPath, 'framedb')
        testCreator = TestCreator(clipFile, 5, framedbPath)
        framedb = testCreator.get_framedb()
        frameExtractorTime.toc()
        # evaluate frames
        caffeTime.tic()
        preddb = frameEvaluater.evaluate(framedb)
        caffeTime.toc()
        # save to pickle
        pickleTime.tic()
        preddbPath = os.path.join(clipOutPath, 'preddb.pkl')
        with open(preddbPath, 'wb') as f:
            pickle.dump(preddb, f)
        pickleTime.toc()
        logging.info("Avg. times: FrameExtractor {}, Caffe: {}, Pickle: {}".format(
            frameExtractorTime.average_time, caffeTime.average_time, pickleTime.average_time))

    logging.info("Total times: FrameExtractor {}, Caffe: {}, Pickle: {}".format(
        frameExtractorTime.total_time, caffeTime.total_time, pickleTime.total_time))
    # from IPython import embed; embed()
