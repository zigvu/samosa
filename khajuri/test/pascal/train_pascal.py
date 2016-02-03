#!/usr/bin/env python

import logging
import json

import _init_paths
from pascal_creator import PascalCreator

from chia.train.train_config import TrainConfig
from chia.train.train_templater import TrainTemplater
from chia.train.train_dataset import TrainDataset
from chia.train.fine_tuner import FineTuner
from tools.files.file_utils import FileUtils

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start training.')

    with open('voc_config_train.json') as f:
        configHash = json.load(f)

    config = TrainConfig(configHash)
    config.reset_folders()
    templater = TrainTemplater()

    dataCreator = PascalCreator('trainval', '2007', '/home/ubuntu/VOC/VOCdevkit')
    dataset = TrainDataset(dataCreator)
    fineTuner = FineTuner(dataset)
    fineTuner.train()

    # from IPython import embed; embed()
