"""Test caffe on framedb."""

import logging
from collections import OrderedDict
import numpy as np

from chia.test.bbox_reducer import BboxReducer

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import caffe


class FrameEvaluaterError(Exception):
    pass

class FrameEvaluater(object):
    def __init__(self, gpuId):
        logging.info("Start frame evaluation in GPU: {}".format(gpuId))
        self.bboxReducer = BboxReducer()
        self.setup(gpuId)

    def setup(self, gpuId):
        caffe.set_mode_gpu()
        caffe.set_device(gpuId)
        # boost interface requires explicit conversion to ascii from unicode
        prototxt_test = str(cfg.ZIGVU.FILES.PROTOTXT_TEST)
        caffemodel = str(cfg.ZIGVU.FILES.CURRENT_MODEL)

        self.net = caffe.Net(prototxt_test, caffemodel, caffe.TEST)

        # Warmup on a dummy image
        im = 128 * np.ones((1280, 720, 3), dtype=np.uint8)
        for i in xrange(2):
            _,_,_ = im_detect(self.net, im)

    def evaluate(self, framedb):
        preddb = OrderedDict()
        for fn, im in framedb.iteritems():
            scores, boxes, fc7 = im_detect(self.net, im)
            nms_boxes, fc7_inds, fc7_scores = self.bboxReducer.evaluate(scores, boxes, fc7)
            preddb[fn] = OrderedDict()
            preddb[fn]['nms_boxes'] = nms_boxes
            preddb[fn]['fc7_inds'] = fc7_inds
            preddb[fn]['fc7_scores'] = fc7_scores
        return preddb
