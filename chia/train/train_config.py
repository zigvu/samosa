"""Create config to train zigvu dataset."""

import logging
import os

from chia._init_paths import CHIA_ROOT
from chia.configs.chia_config import chia_cfg
from tools.files.file_utils import FileUtils

from fast_rcnn.config import cfg, cfg_from_file

class TrainConfigError(Exception):
    pass

class TrainConfig(object):
    def __init__(self, config_hash):
        logging.info("Configuring training")
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
        FileUtils.rm_rf(chia_cfg.TRAIN.FOLDERS.ROOT)
        self._create_folders()

    def _set_configs_from_hash(self, config_hash):
        ch = config_hash
        if ch['mode'] != 'train':
            raise TrainConfigError("Supplied config is not a training config")
        cfg.GPU_ID = int(ch['gpu_device_id'])
        # zigvu specific configs
        chia_cfg.TRAIN.ITERATION_ID = ch['iteration_id']
        chia_cfg.TRAIN.PARENT_ITERATION_ID = ch['parent_iteration_id']
        chia_cfg.TRAIN.IS_MAJOR_ITERATION = ch['iteration_type'] == 'major'
        chia_cfg.TRAIN.MAX_ITERS = int(ch['num_caffe_iteration'])
        # add background class with index 0
        chia_cfg.TRAIN.POSITIVE_CLASSES =  ['__background__'] + ch['positive_classes']
        chia_cfg.TRAIN.AVOID_CLASSES = ch['avoid_classes']
        # output folder
        chia_cfg.TRAIN.FOLDERS.OUTPUT = ch['output_folder']

    def _create_folders(self):
        chia_cfg.TRAIN.FOLDERS.ROOT = os.path.join(
                chia_cfg.TRAIN.FOLDERS.OUTPUT, chia_cfg.TRAIN.ITERATION_ID)
        chia_cfg.TRAIN.FOLDERS.CACHE = os.path.join(chia_cfg.TRAIN.FOLDERS.ROOT, 'cache')
        FileUtils.mkdir_p(chia_cfg.TRAIN.FOLDERS.CACHE)
        chia_cfg.TRAIN.FOLDERS.PROTOTXT = os.path.join(chia_cfg.TRAIN.FOLDERS.ROOT, 'prototxt')
        FileUtils.mkdir_p(chia_cfg.TRAIN.FOLDERS.PROTOTXT)
        chia_cfg.TRAIN.FOLDERS.MODEL = os.path.join(chia_cfg.TRAIN.FOLDERS.ROOT, 'model')
        FileUtils.mkdir_p(chia_cfg.TRAIN.FOLDERS.MODEL)
        self._create_files()

    def _create_files(self):
        chia_cfg.TRAIN.FILES.PROTOXT_SOLVER = os.path.join(
                chia_cfg.TRAIN.FOLDERS.PROTOTXT, 'solver.prototxt')
        chia_cfg.TRAIN.FILES.PROTOTXT_TRAIN = os.path.join(
                chia_cfg.TRAIN.FOLDERS.PROTOTXT, 'train.prototxt')

        chia_cfg.TRAIN.FILES.OUTPUT_MODEL = os.path.join(chia_cfg.TRAIN.FOLDERS.MODEL, chia_cfg.TRAIN.ITERATION_ID)
        chia_cfg.TRAIN.FILES.PARENT_MODEL = os.path.join(chia_cfg.TRAIN.FOLDERS.MODEL, chia_cfg.TRAIN.PARENT_ITERATION_ID)
