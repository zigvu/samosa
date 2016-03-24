#!/usr/bin/env python

import logging
import argparse
import json

import _init_paths
from chia.configs.chia_config import chia_cfg

from chia.datasets.clips_to_frames import ClipsToFrames
from chia.datasets.train_creator import TrainCreator
from chia.train.train_config import TrainConfig
from chia.train.train_templater import TrainTemplater
from chia.train.train_dataset import TrainDataset
from chia.train.fine_tuner import FineTuner

from tools.files.file_utils import FileUtils

from fast_rcnn.config import cfg

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train on zigvu local dataset')
    parser.add_argument('--config_file', dest='config_file',
                        help='Path to config file for training', required=True)
    parser.add_argument('--parent_model', dest='parent_model',
                        help='Base caffe model for fine tuning', required=True)
    parser.add_argument('--annotation_folder', dest='annotation_folder',
                        help='Path to annotation folder', required=True)
    parser.add_argument('--clip_folder', dest='clip_folder',
                        help='Path to clip folder', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start training.')

    args = parse_args()

    with open(args.config_file) as f:
        configHash = json.load(f)

    config = TrainConfig(configHash)
    # config.reset_folders()
    templater = TrainTemplater()

    FileUtils.symlink(args.parent_model, chia_cfg.TRAIN.FILES.PARENT_MODEL)

    frameFolder = args.annotation_folder
    clipsToFrames = ClipsToFrames(args.clip_folder, args.annotation_folder, frameFolder)
    clipsToFrames.extract
    dataCreator = TrainCreator(args.annotation_folder, frameFolder)
    dataset = TrainDataset(dataCreator)
    fineTuner = FineTuner(dataset)
    fineTuner.train()

    # from IPython import embed; embed()
