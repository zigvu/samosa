"""Create config to train zigvu dataset."""

import logging
import os
from easydict import EasyDict as edict

from chia._init_paths import CHIA_ROOT
from tools.file_utils import FileUtils

from fast_rcnn.config import cfg, cfg_from_file

class TestConfigError(Exception):
    pass

class TestConfig(object):
    def __init__(self, config_hash):
        logging.info("Configuring testing")
        # create new config variables
        self._create_zigvu_configs()
        # set defaults from file
        cfg_file = '{}/experiments/cfgs/zigvu_end2end.yml'.format(CHIA_ROOT)
        cfg_from_file(cfg_file)
        # set values from config_hash
        self._set_configs_from_hash(config_hash)
        # create folders
        self._create_folders()
        # log config for debug
        logging.debug('Config from Rasbari: %s', config_hash)
        logging.debug('Current config: %s', cfg)

    def reset_folders(self):
        FileUtils.rm_rf(cfg.ZIGVU.FOLDERS.ROOT)
        self._create_folders()

    def _set_configs_from_hash(self, config_hash):
        ch = config_hash
        if ch['mode'] != 'test':
            raise TrainConfigError("Supplied config is not a testing config")
        cfg.GPU_ID = int(ch['gpu_device_id'])

        # zigvu specific configs
        cfg.ZIGVU.ITERATION_ID = ch['iteration_id']
        # add background class with index 0
        cfg.ZIGVU.POSITIVE_CLASSES =  ['__background__'] + ch['positive_classes']
        cfg.ZIGVU.AVOID_CLASSES = ch['avoid_classes']

    def _create_zigvu_configs(self):
        cfg.ZIGVU = edict()
        cfg.ZIGVU.ITERATION_ID = None
        cfg.ZIGVU.POSITIVE_CLASSES = []
        cfg.ZIGVU.AVOID_CLASSES = []
        cfg.ZIGVU.FOLDERS = edict()
        cfg.ZIGVU.FILES = edict()
        cfg.ZIGVU.MAX_ITERS = 100

    def _create_folders(self):
        cfg.ZIGVU.FOLDERS.ROOT = os.path.join('/tmp', cfg.ZIGVU.ITERATION_ID)
        cfg.ZIGVU.FOLDERS.CACHE = os.path.join(cfg.ZIGVU.FOLDERS.ROOT, 'cache')
        FileUtils.mkdir_p(cfg.ZIGVU.FOLDERS.CACHE)
        cfg.ZIGVU.FOLDERS.PROTOTXT = os.path.join(cfg.ZIGVU.FOLDERS.ROOT, 'prototxt')
        FileUtils.mkdir_p(cfg.ZIGVU.FOLDERS.PROTOTXT)
        cfg.ZIGVU.FOLDERS.MODEL = os.path.join(cfg.ZIGVU.FOLDERS.ROOT, 'model')
        FileUtils.mkdir_p(cfg.ZIGVU.FOLDERS.MODEL)
        self._create_files()

    def _create_files(self):
        cfg.ZIGVU.FILES.PROTOTXT_TEST = os.path.join(cfg.ZIGVU.FOLDERS.PROTOTXT, 'test.prototxt')
        cfg.ZIGVU.FILES.CURRENT_MODEL = os.path.join(cfg.ZIGVU.FOLDERS.MODEL, cfg.ZIGVU.ITERATION_ID)
