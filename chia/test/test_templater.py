"""Create training prototxts based on templates."""

import logging

from chia._init_paths import CHIA_ROOT
from tools.files.file_changer import FileChanger
from tools.files.file_utils import FileUtils # TODO: remove

from fast_rcnn.config import cfg

class TestTemplaterError(Exception):
    pass

class TestTemplater(object):
    def __init__(self):
        logging.info("Templating testing prototxts")
        self._template_prototxt()
        # TODO: this should ideally come from nimki
        FileUtils.symlink(
            '/home/ubuntu/samosa/py-faster-rcnn/data/imagenet_models/ZF.v2.caffemodel', cfg.ZIGVU.FILES.CURRENT_MODEL
        )

    def _template_prototxt(self):
        self._template_test_prototxt()

    def _template_test_prototxt(self):
        template = '{}/models/ZF/zigvu_end2end/test.prototxt'.format(CHIA_ROOT)
        num_classes = len(cfg.ZIGVU.POSITIVE_CLASSES)
        replace = {
            'ZIGVU_NUM_CLASSES': num_classes, # num_classes
            'ZIGVU_BBOX_PRED_OUTPUT': num_classes * 4 # num_classes * 4
        }
        FileChanger.regex(template, replace, cfg.ZIGVU.FILES.PROTOTXT_TEST)
