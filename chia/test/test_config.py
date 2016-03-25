"""Create config to train zigvu dataset."""

import logging
import os
from easydict import EasyDict as edict

from chia._init_paths import CHIA_ROOT
from chia.configs.chia_config import chia_cfg
from tools.files.file_utils import FileUtils

from fast_rcnn.config import cfg, cfg_from_file

class TestConfigError(Exception):
    pass

class TestConfig(object):
    def __init__(self, config_hash):
        logging.info("Configuring testing")
        cfg.IS_ZIGVU_RUN = True
        # override fast_rcnn settings
        cfg_file = os.path.join(CHIA_ROOT, 'configs/experiments/cfgs/zigvu_end2end.yml')
        cfg_from_file(cfg_file)
        # set values from config_hash
        self._set_configs_from_hash(config_hash)
        # create folders
        self._create_folders()
        # log config for debug
        logging.debug('Config from Rasbari: %s', config_hash)
        logging.debug('Current fast_rcnn config: %s', cfg)
        logging.debug('Current chia config: %s', chia_cfg)

    def reset_folders(self):
        FileUtils.rm_rf(chia_cfg.TEST.FOLDERS.ROOT)
        self._create_folders()

    def _set_configs_from_hash(self, config_hash):
        ch = config_hash
        if ch['mode'] != 'test':
            raise TestConfigError("Supplied config is not a testing config")
        cfg.GPU_ID = int(ch['gpu_device_id'])

        # zigvu specific configs
        chia_cfg.TEST.CHIA_MODEL_ID = ch['chia_model_id']
        # add background class with index 0
        chia_cfg.TEST.POSITIVE_CLASSES =  ['__background__'] + ch['positive_classes']
        chia_cfg.TEST.AVOID_CLASSES = ch['avoid_classes']

    def _create_folders(self):
        chia_cfg.TEST.FOLDERS.ROOT = os.path.join(chia_cfg.TEST.FOLDERS.OUTPUT, chia_cfg.TEST.CHIA_MODEL_ID)
        chia_cfg.TEST.FOLDERS.CACHE = os.path.join(chia_cfg.TEST.FOLDERS.ROOT, 'cache')
        FileUtils.mkdir_p(chia_cfg.TEST.FOLDERS.CACHE)
        chia_cfg.TEST.FOLDERS.PROTOTXT = os.path.join(chia_cfg.TEST.FOLDERS.ROOT, 'prototxt')
        FileUtils.mkdir_p(chia_cfg.TEST.FOLDERS.PROTOTXT)
        chia_cfg.TEST.FOLDERS.MODEL = os.path.join(chia_cfg.TEST.FOLDERS.ROOT, 'model')
        FileUtils.mkdir_p(chia_cfg.TEST.FOLDERS.MODEL)
        self._create_files()

    def _create_files(self):
        chia_cfg.TEST.FILES.PROTOTXT_TEST = os.path.join(chia_cfg.TEST.FOLDERS.PROTOTXT, 'test.prototxt')
        chia_cfg.TEST.FILES.CURRENT_MODEL = os.path.join(chia_cfg.TEST.FOLDERS.MODEL, chia_cfg.TEST.CHIA_MODEL_ID)
