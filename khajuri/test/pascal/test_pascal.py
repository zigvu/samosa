#!/usr/bin/env python

import logging
import argparse
import json

import _init_paths
from pascal_creator import PascalCreator

from chia.test.test_config import TestConfig
from chia.test.test_templater import TestTemplater

from fast_rcnn.config import cfg

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Test on zigvu model on clip')
    parser.add_argument('--config_file', dest='config_file',
                        help='Path to config file for training', required=True)
    parser.add_argument('--test_model', dest='test_model',
                        help='Caffe model for testing', required=True)


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

    # from IPython import embed; embed()
