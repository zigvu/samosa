import os
import pickle
import json
import logging
from collections import OrderedDict
import numpy as np

from khajuri.multi.task import Task
from tools.files.file_utils import FileUtils


class FileSaverError(Exception):
    pass

class FileSaver(Task):
    def __init__(self):
        Task.__init__(self, 'FileSaver')

    def start(self):
        pass

    def process(self, clip):
        logging.debug('{}: process clip: {}'.format(self.taskName, clip.clip_id))
        FileUtils.mkdir_p(clip.result_path['base_path'])
        with open(clip.result_path['pickle'], 'wb') as f:
            pickle.dump(clip, f)
        clipJson = self.clip_to_json(clip)
        with open(clip.result_path['json'], 'w') as f:
            json.dump(clipJson, f)
        return clip

    # returns JSON to dump
    def clip_to_json(self, clip):
        preddb = clip.preddb
        confdb = clip.confdb
        clipJson = OrderedDict()
        clipJson['meta'] = OrderedDict()
        clipJson['meta']['clip_id'] = int(clip.clip_id)
        if clip.clip_eval_details != None:
            clipJson['meta']['cap_eval_id'] = str(clip.clip_eval_details.message['capEvalId'])
        clipJson['meta']['scale'] = 1.0 # this doesn't change per RCNN
        # format as per rails requirement
        data = OrderedDict()
        numCls = len(preddb[preddb.keys()[0]]['nms_boxes'])
        for fn in preddb.keys():
            # format:
            # {cls1Idx: {cls2Idx: {cls1BboxIdx: [intersects_cls2BboxIdx,] }, }, }
            inters = OrderedDict()
            # format:
            # {cls1Idx: {cls2Idx: {cls1BboxIdx: [interThresh_cls2BboxIdx,] }, }, }
            thresh = OrderedDict()
            # format:
            # {cls1Idx: {bboxIdx: [x0,y0,x3,y3,score,zdist], }, }
            nms_boxes = OrderedDict()
            for cls1Idx in xrange(numCls):
                inters[cls1Idx] = OrderedDict()
                thresh[cls1Idx] = OrderedDict()
                for cls2Idx in xrange(numCls):
                    if cls1Idx == cls2Idx:
                        continue
                    inters[cls1Idx][cls2Idx] = OrderedDict()
                    thresh[cls1Idx][cls2Idx] = OrderedDict()
                    conf = confdb[fn][cls1Idx][cls2Idx]
                    interClsIdx = np.argwhere(conf > 0.001)
                    if len(conf) > 0 and len(interClsIdx) > 1:
                        for bb in interClsIdx:
                            interVal = bb[1]
                            threshVal = round(conf[bb[0]][bb[1]], 3)
                            if bb[0] in inters[cls1Idx][cls2Idx]:
                                inters[cls1Idx][cls2Idx][bb[0]].append(interVal)
                                thresh[cls1Idx][cls2Idx][bb[0]].append(threshVal)
                            else:
                                inters[cls1Idx][cls2Idx][bb[0]] = [interVal]
                                thresh[cls1Idx][cls2Idx][bb[0]] = [threshVal]
                # done for cls2Idx
                nms_boxes[cls1Idx] = OrderedDict()
                nmsBbox = preddb[fn]['nms_boxes'][cls1Idx]
                for idx in xrange(len(nmsBbox)):
                    nms_boxes[cls1Idx][idx] = np.round(nmsBbox[idx].astype(np.float64), 3).tolist()
            # done for cls1Idx
            data[fn] = OrderedDict()
            data[fn]['nms_boxes'] = nms_boxes
            data[fn]['inters'] = inters
            data[fn]['thresh'] = thresh
        clipJson['data'] = data
        return clipJson
