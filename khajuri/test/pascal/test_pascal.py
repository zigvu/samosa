#!/usr/bin/env python

import logging
import json

import _init_paths
from pascal_creator import PascalCreator

from chia.test.test_config import TestConfig
from chia.test.test_templater import TestTemplater

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Start training.')

    with open('voc_config_test.json') as f:
        configHash = json.load(f)

    config = TestConfig(configHash)
    # config.reset_folders()
    templater = TestTemplater()

    # from IPython import embed; embed()
