"""Combine predicted bboxes."""

import logging
import numpy as np

from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms


class BboxReducerError(Exception):
    pass

class BboxReducer(object):
    def __init__(self):
        self.num_classes = len(cfg.ZIGVU.POSITIVE_CLASSES)
        self.thresh = cfg.ZIGVU.detection_thresh
        self.max_per_image = cfg.ZIGVU.nms_box_max_per_image

    def evaluate(self, scores, boxes, fc7):
        all_fc7_inds = []
        nms_boxes = [[] for j in xrange(self.num_classes - 1)]
        fc7_inds = [[] for j in xrange(self.num_classes - 1)]
        for j in xrange(1, self.num_classes):
            # limit number of boxes through threshold and max_per_image
            inds = np.where(scores[:, j] > self.thresh)[0]
            thresh_scores = scores[inds, j]
            thresh_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-thresh_scores)[:self.max_per_image]
            cls_scores = thresh_scores[top_inds]
            cls_boxes = thresh_boxes[top_inds, :]
            # perform NMS
            dets = np.hstack(
                (cls_boxes, cls_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
            if len(dets) > 0:
                keep = nms(dets, self.thresh)
                if len(keep) > 0:
                    nms_boxes[j-1] = dets[keep, :].copy()
                    # keep track of fc7 scores
                    orig_inds = inds[top_inds[keep]]
                    for idx in orig_inds:
                        if idx not in all_fc7_inds:
                            all_fc7_inds.append(idx)
                            fc7_inds[j-1].append(len(all_fc7_inds) - 1)
                        else:
                            fc7_inds.append(all_fc7_inds.index(idx))
                    # end for idx
                # end if
            # end if len(dets)
        # end for class
        fc7_scores = fc7[all_fc7_inds, :]
        return nms_boxes, fc7_inds, fc7_scores
