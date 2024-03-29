import logging
from collections import OrderedDict
import numpy as np

class FindConfusions(object):
    def __init__(self, scoreThreshs, interThreshs):
        self.scoreThreshs = scoreThreshs
        self.interThreshs = interThreshs

    def confusions(self, preddb):
        numCls = len(preddb[preddb.keys()[0]]['nms_boxes'])
        confdb = OrderedDict()
        for fn, pred in preddb.iteritems():
            nmsBoxes = pred['nms_boxes']
            confMat = self._confusion_mat(nmsBoxes)
            confInds = OrderedDict()
            for sth1 in self.scoreThreshs:
                confInds[sth1] = OrderedDict()
                for sth2 in self.scoreThreshs:
                    confInds[sth1][sth2] = OrderedDict()
                    for ith in self.interThreshs:
                        confInds[sth1][sth2][ith] = OrderedDict()
                        for cls1Idx in xrange(numCls):
                            confInds[sth1][sth2][ith][cls1Idx] = OrderedDict()
                            for cls2Idx in xrange(numCls):
                                if cls1Idx != cls2Idx:
                                    confInds[sth1][sth2][ith][cls1Idx][cls2Idx] = self._confusion_inds(
                                        nmsBoxes, confMat, sth1, sth2, ith, cls1Idx, cls2Idx)
                            # end cls2Idx
                        # end cls1Idx
                    # end ith
                # end sth2
            # end sth1
            confdb[fn] = OrderedDict()
            confdb[fn]['conf_mat'] = confMat
            confdb[fn]['conf_inds'] = confInds
        return confdb

    def _percentage_intersect(self, a, b):
        """Area intersection of `a` and `b` as fraction of area of `a`"""
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        inter = 0
        if dx > 0 and dy > 0:
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            inter = dx * dy * 1.0 / area_a
        return inter

    def _intersect(self, bbAs, bbBs):
        """Intersection between bbox sets as percentage of area of bbAs"""
        inter = []
        if len(bbAs) == 0 or len(bbBs) == 0:
            return inter
        inter = np.zeros([np.shape(bbAs)[0],np.shape(bbBs)[0]])
        for idxA, a in enumerate(bbAs):
            for idxB, b in enumerate(bbBs):
                inter[idxA][idxB] = self._percentage_intersect(a,b)
        return inter

    def _confusion_mat(self, nms_boxes):
        """Confusion of nms-ed bboxes"""
        confMat = [[[] for i in xrange(0, len(nms_boxes))] for i in xrange(0, len(nms_boxes))]
        for idxCls1, bboxesCls1 in enumerate(nms_boxes):
            for idxCls2, bboxesCls2 in enumerate(nms_boxes):
                confMat[idxCls1][idxCls2] = self._intersect(bboxesCls1, bboxesCls2)
        return confMat

    def _confusion_inds(self, nms_boxes, confMat, cls1ScThresh, cls2ScThresh, inThresh, cls1Idx, cls2Idx):
        """Indices of nms_boxes that have above threshold confusion"""
        nmsConfIdxs = []
        for i, cls1Box in enumerate(nms_boxes[cls1Idx]):
            for j, cls2Box in enumerate(nms_boxes[cls2Idx]):
                if confMat[cls1Idx][cls2Idx][i][j] > inThresh and \
                        cls1Box[4] > cls1ScThresh and cls2Box[4] > cls2ScThresh:
                    nmsConfIdxs.append(i)
        return nmsConfIdxs
