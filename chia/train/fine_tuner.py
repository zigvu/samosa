"""Train caffe on imdb/roidb."""

import logging
import numpy as np

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.train import get_training_roidb, train_net
import caffe


class FineTunerError(Exception):
    pass

class FineTuner(object):
    def __init__(self, dataset):
        logging.info("Start caffe training")
        self.imdb = dataset
        # TODO: logic for major/minor iteration
        self._max_iters = cfg.ZIGVU.MAX_ITERS

    def train(self):
        self.imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        logging.debug('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        self.roidb = get_training_roidb(self.imdb)
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
        # set up caffe
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
        # boost interface requires explicit conversion to ascii from unicode
        prototxt_solver = str(cfg.ZIGVU.FILES.PROTOXT_SOLVER)
        output_dir = str(cfg.ZIGVU.FOLDERS.MODEL)
        parent_model = str(cfg.ZIGVU.FILES.PARENT_MODEL)
        logging.debug('Output will be saved to `{:s}`'.format(output_dir))

        train_net(prototxt_solver, self.roidb, output_dir,
                  pretrained_model=parent_model,
                  max_iters=self._max_iters)
