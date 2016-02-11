import os
import logging
import json
from easydict import EasyDict as edict

"""Read annotations from file"""

class FileReader(object):
    def __init__(self, annotation_file):
        self._annotation_file = annotation_file
        assert os.path.exists(annotation_file), \
            'Annotation file does not exist: {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            self._anno = edict(json.load(f))
        logging.debug("Read annotation file {}".format(annotation_file))

    def get_frame_file(self, output_folder):
        frameFile = "{}/{}.png".format(output_folder, self._anno.frame_number)
        assert os.path.exists(frameFile), \
            'Corresponding frame file does not exist: {}'.format(self._annotation_file)
        return frameFile

    def get_image_index(self):
        return "{}_{}".format(self._anno.clip_id, self._anno.frame_number)

    def get_classes(self):
        return self._anno.annotations.keys()

    def get_bboxes(self, cls):
        bboxes = []
        if cls in self._anno.annotations.keys():
            for bbox in self._anno.annotations[cls]:
                bboxes.append(self._bbox_normalize(bbox, cls))
        return bboxes

    def is_minor_iteration(self):
        return self._anno.is_minor == True

    def _bbox_normalize(self, bbox, cls):
        x0 = min(bbox.x0, bbox.x1, bbox.x2, bbox.x3)
        y0 = min(bbox.y0, bbox.y1, bbox.y2, bbox.y3)
        x2 = max(bbox.x0, bbox.x1, bbox.x2, bbox.x3)
        y2 = max(bbox.y0, bbox.y1, bbox.y2, bbox.y3)
        return edict({'x0': x0, 'y0': y0, 'x2': x2, 'y2': y2, 'cls': cls})
