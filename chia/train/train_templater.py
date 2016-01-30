"""Create training prototxts based on templates."""

import logging

from chia._init_paths import CHIA_ROOT
from tools.file_regexer import FileRegexer
from tools.file_utils import FileUtils # TODO: remove

from fast_rcnn.config import cfg

class TrainTemplaterError(Exception):
    pass

class TrainTemplater(object):
    def __init__(self):
        logging.info("Templating training prototxts")
        self._template_prototxt()
        # TODO: this should ideally come from nimki
        FileUtils.symlink(
            '/home/ubuntu/samosa/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel', cfg.ZIGVU.FILES.PARENT_MODEL
        )

    def _template_prototxt(self):
        self._template_solver_prototxt()
        self._template_train_prototxt()

    def _template_solver_prototxt(self):
        template = '{}/models/ZF/zigvu_end2end/solver.prototxt'.format(CHIA_ROOT)
        replace = {
            'ZIGVU_TRAIN_FILE': cfg.ZIGVU.FILES.PROTOTXT_TRAIN
        }
        FileRegexer.regex(template, replace, cfg.ZIGVU.FILES.PROTOXT_SOLVER)

    def _template_train_prototxt(self):
        template = '{}/models/ZF/zigvu_end2end/train.prototxt'.format(CHIA_ROOT)
        num_classes = len(cfg.ZIGVU.POSITIVE_CLASSES)
        replace = {
            'ZIGVU_NUM_CLASSES': num_classes, # num_classes
            'ZIGVU_BBOX_PRED_OUTPUT': num_classes * 4 # num_classes * 4
        }
        FileRegexer.regex(template, replace, cfg.ZIGVU.FILES.PROTOTXT_TRAIN)
