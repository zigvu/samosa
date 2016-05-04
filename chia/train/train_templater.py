"""Create training prototxts based on templates."""

import logging

from chia._init_paths import CHIA_ROOT
from chia.configs.chia_config import chia_cfg
from tools.files.file_changer import FileChanger

from fast_rcnn.config import cfg

class TrainTemplaterError(Exception):
    pass

class TrainTemplater(object):
    def __init__(self, is_temp_proto = False):
        logging.info("Templating training prototxts")
        self.isTempProto = is_temp_proto
        self._template_prototxt()

    def _template_prototxt(self):
        self._template_solver_prototxt()
        self._template_train_prototxt()

    def _template_solver_prototxt(self):
        template = '{}/configs/models/ZF/zigvu_end2end/solver.prototxt'.format(CHIA_ROOT)
        replace = {
            'ZIGVU_TRAIN_FILE': chia_cfg.TRAIN.FILES.PROTOTXT_TRAIN
        }
        FileChanger.regex(template, replace, chia_cfg.TRAIN.FILES.PROTOXT_SOLVER)

    def _template_train_prototxt(self):
        template = '{}/configs/models/ZF/zigvu_end2end/train.prototxt'.format(CHIA_ROOT)
        num_classes = len(chia_cfg.TRAIN.POSITIVE_CLASSES)
        bboxPredName = 'bbox_pred'
        clsScoreName = 'cls_score'
        if self.isTempProto:
            bboxPredName = 'bbox_pred_TEMP'
            clsScoreName = 'cls_score_TEMP'
        replace = {
            'ZIGVU_NUM_CLASSES': num_classes,
            'ZIGVU_BBOX_PRED_OUTPUT': num_classes * 4,
            'ZIGVU_BBOX_PRED_NAME': bboxPredName,
            'ZIGVU_CLS_SCORE_NAME': clsScoreName
        }
        FileChanger.regex(template, replace, chia_cfg.TRAIN.FILES.PROTOTXT_TRAIN)
