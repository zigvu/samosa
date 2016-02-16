"""Create training prototxts based on templates."""

import logging

from chia._init_paths import CHIA_ROOT
from tools.files.file_changer import FileChanger

from fast_rcnn.config import cfg

class TestTemplaterError(Exception):
    pass

class TestTemplater(object):
    def __init__(self):
        logging.info("Templating testing prototxts")
        self._template_prototxt()

    def _template_prototxt(self):
        self._template_test_prototxt()

    def _template_test_prototxt(self):
        template = '{}/models/ZF/zigvu_end2end/test.prototxt'.format(CHIA_ROOT)
        num_classes = len(cfg.ZIGVU.POSITIVE_CLASSES)
        replace = {
            'ZIGVU_NUM_CLASSES': num_classes,
            'ZIGVU_BBOX_PRED_OUTPUT': num_classes * 4
        }
        FileChanger.regex(template, replace, cfg.ZIGVU.FILES.PROTOTXT_TEST)
