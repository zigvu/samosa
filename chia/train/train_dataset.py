"""Dataset that conforms to py-faster-caffe requirements."""

import os
import logging
import numpy as np
import scipy.sparse
import cPickle

from chia.configs.chia_config import chia_cfg

from fast_rcnn.config import cfg, cfg_from_file
import datasets
import datasets.imdb

class TrainDatasetError(Exception):
    pass

class TrainDataset(datasets.imdb):
    def __init__(self, data_creator):
        logging.info("Creating training dataset")
        datasets.imdb.__init__(self, chia_cfg.TRAIN.CHIA_MODEL_ID)
        self._data_creator = data_creator
        self._classes = chia_cfg.TRAIN.POSITIVE_CLASSES
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # caching
        self._gt_roidb = None
        self._dataset_cache_file = os.path.join(chia_cfg.TRAIN.FOLDERS.CACHE, self.name + '_dataset.pkl')
        self._roidb_cache_file = os.path.join(chia_cfg.TRAIN.FOLDERS.CACHE, self.name + '_gt_roidb.pkl')
        self._data, self._image_index = self._cache_dataset()

    def _cache_dataset(self):
        """
        Caches dataset so that subsequent calls are fast.
        """
        # if exists in cache, serve
        if os.path.exists(self._dataset_cache_file):
            with open(self._dataset_cache_file, 'rb') as fid:
                dataset = cPickle.load(fid)
            logging.debug('{} dataset loaded from {}'.format(self.name, self._dataset_cache_file))
            return dataset['data'], dataset['image_index']
        # create and save data
        dataset = self._data_creator.getDataset()
        with open(self._dataset_cache_file, 'wb') as fid:
            cPickle.dump(dataset, fid, cPickle.HIGHEST_PROTOCOL)
        logging.debug('wrote dataset to {}'.format(self._dataset_cache_file))
        return dataset['data'], dataset['image_index']

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_name = self._image_index[i]
        return self._data[image_name]['image_filename']

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # if exists in memory, serve
        if self._gt_roidb != None:
            return self._gt_roidb
        # if exists in cache, serve
        if os.path.exists(self._roidb_cache_file):
            with open(self._roidb_cache_file, 'rb') as fid:
                self._gt_roidb = cPickle.load(fid)
            logging.debug('{} gt roidb loaded from {}'.format(self.name, self._roidb_cache_file))
            return self._gt_roidb
        # create and save data
        self._gt_roidb = []
        for image_name in self._image_index:
            self._gt_roidb.append(self._data_to_gt_roidb(self._data[image_name]))
        # [self._data_to_gt_roidb(d) for d in self._data]
        with open(self._roidb_cache_file, 'wb') as fid:
            cPickle.dump(self._gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        logging.debug('wrote gt roidb to {}'.format(self._roidb_cache_file))
        # return data
        return self._gt_roidb

    def _data_to_gt_roidb(self, data):
        """
        Load bboxes to db.
        """
        objs = data['bboxes']
        num_objs = len(objs)

        if num_objs < 1:
            raise TrainDatasetError("No bboxes for image: {}".format(data['image_filename']))

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            boxes[ix, :] = [obj['x0'], obj['y0'], obj['x2'], obj['y2']]
            cls = self._class_to_ind[obj['cls']]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        # Add avoid bboxes
        avoids = data['avoid_bboxes']
        num_avoids = len(avoids)
        avoid_boxes = np.zeros((num_avoids, 4), dtype=np.uint16)
        for ix, obj in enumerate(avoids):
            avoid_boxes[ix, :] = [obj['x0'], obj['y0'], obj['x2'], obj['y2']]

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'avoid_boxes': avoid_boxes,
                'image_filename' : data['image_filename'],
                'is_minor_iteration' : data['is_minor_iteration']}
