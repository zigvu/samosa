import os
import logging
from collections import OrderedDict

from chia.configs.chia_config import chia_cfg
from chia.datasets.annotation_reader import AnnotationReader
from tools.files.file_changer import FileChanger

from fast_rcnn.config import cfg


class TrainCreatorError(Exception):
    pass

class TrainCreator(object):
    def __init__(self, annotation_path, frame_path):
        self._annotation_path = annotation_path
        self._frame_path = frame_path

        self._clip_ids = []

        # format:
        # {
        #  image_index: [:image_name, ],
        #  data: {:image_name => {image_filename:, bboxes: [{x1:, y1: x2:, y2:, cls:}, ]}, }
        # }
        self._dataset = None
        self._ensure_files_exit()

    def _ensure_files_exit(self):
        assert os.path.exists(self._annotation_path), \
                'Annotation path does not exist: {}'.format(self._annotation_path)
        assert os.path.exists(self._frame_path), \
                'Frame path does not exist: {}'.format(self._frame_path)
        self._clip_ids = [clipId for clipId in os.listdir(self._annotation_path)
                    if os.path.isdir(os.path.join(self._annotation_path, clipId))]
        for clipId in self._clip_ids:
            # ensure all frames are extracted
            origFnsFile = os.path.join(self._annotation_path, clipId, 'frame_numbers.txt')
            extrFnsFile = os.path.join(self._frame_path, clipId, 'extracted_frame_numbers.txt')
            origFns = FileChanger.int_array_reader(origFnsFile)
            extrFns = FileChanger.int_array_reader(extrFnsFile)
            if sorted(origFns) != sorted(extrFns):
                raise TrainCreatorError("Extracted frame mismatch for clip id {}".format(clipId))


    def getDataset(self):
        """
        Return combined data
        """
        if self._dataset == None:
            image_index, data = self._load_data()
            self._dataset = OrderedDict({
                'image_index': image_index,
                'data': data
            })
        return self._dataset

    def _load_data(self):
        """
        Loads all data from files.
        """
        image_index = []
        data = OrderedDict()
        for clipId in self._clip_ids:
            origFnsFile = os.path.join(self._annotation_path, clipId, 'frame_numbers.txt')
            origFns = FileChanger.int_array_reader(origFnsFile)
            frameOutputFolder = os.path.join(self._frame_path, clipId, 'frames')
            for fn in origFns:
                annoFile = os.path.join(
                    self._annotation_path, clipId, 'annotations', '{}.json'.format(fn)
                )
                annoReader = AnnotationReader(annoFile)
                index = annoReader.get_image_index()
                d = self._load_annotation(annoReader, frameOutputFolder)
                if len(d['bboxes']) > 0:
                    image_index.append(index)
                    data[index] = d
        return image_index, data

    def _load_annotation(self, annoReader, frameOutputFolder):
        """
        Load image and bounding boxes info
        """
        # Load object bounding boxes into a data frame.
        bboxes = []
        for cls in chia_cfg.TRAIN.POSITIVE_CLASSES:
            for bbox in annoReader.get_bboxes(cls):
                bboxes.append(bbox)

        # Load avoid bounding boxes into a data frame.
        avoid_bboxes = []
        for cls in chia_cfg.TRAIN.AVOID_CLASSES:
            for bbox in annoReader.get_bboxes(cls):
                avoid_bboxes.append(bbox)

        return OrderedDict({
            'image_filename': annoReader.get_frame_file(frameOutputFolder),
            'bboxes': bboxes,
            'avoid_bboxes': avoid_bboxes,
            'is_minor_iteration': annoReader.is_minor_iteration()
        })
