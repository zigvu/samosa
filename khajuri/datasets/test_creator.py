import logging
from collections import OrderedDict
import cv2

from tools.files.file_utils import FileUtils
from tools.frames.frame_extractor import FrameExtractor

class TestCreatorError(Exception):
    pass

class TestCreator(object):
    def __init__(self, clip_path, fps, temp_path):
        self.clip_path = clip_path
        self.fps = fps
        self.temp_path = temp_path
        FileUtils.rm_rf(self.temp_path)
        FileUtils.mkdir_p(self.temp_path)

    def get_framedb(self):
        logging.debug("Extract framedb from clip")
        # extract frames
        frameExtractor = FrameExtractor(self.clip_path, self.temp_path)
        fnFileMap = frameExtractor.extract_sequential(self.fps)
        framedb = OrderedDict()
        for fn, fileName in fnFileMap.iteritems():
            framedb[fn] = cv2.imread(fileName)
        # clean up
        FileUtils.rm_rf(self.temp_path)
        return framedb
