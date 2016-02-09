import os
import logging
import xml.dom.minidom as minidom
from collections import OrderedDict

import _init_paths
from fast_rcnn.config import cfg

"""File to load Pascal dataset as if it was from Rasbari"""

class PascalCreator(object):
    def __init__(self, image_set, year, devkit_path):
        self._year = year
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)

        # format:
        # {
        #  image_index: [:image_name, ],
        #  data: {:image_name => {image_filename:, bboxes: [{x1:, y1: x2:, y2:, cls:}, ]}, }
        # }
        self._dataset = None

        self.config = {
            'use_diff' : False
        }

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)

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
        for index in self._load_image_set_index():
            d = self._load_pascal_annotation(index)
            if len(d['bboxes']) > 0:
                image_index.append(index)
                data[index] = d
        return image_index, data

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        image_filename = self._image_path_from_index(index)
        annotation_filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(annotation_filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(annotation_filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs
                             if int(get_data_from_tag(obj, 'difficult')) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects' \
                    .format(len(objs) - len(non_diff_objs))
            objs = non_diff_objs

        # Load object bounding boxes into a data frame.
        bboxes = []
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x0 = float(get_data_from_tag(obj, 'xmin')) - 1
            y0 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            cls = str(get_data_from_tag(obj, "name")).lower().strip()
            if cls in cfg.ZIGVU.POSITIVE_CLASSES:
                bboxes.append(OrderedDict({'x0': x0, 'y0': y0, 'x2': x2, 'y2': y2, 'cls': cls}))

        return OrderedDict({
            'image_filename': image_filename,
            'bboxes': bboxes,
            'avoid_bboxes': []
        })
