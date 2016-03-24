import os
import logging
from collections import OrderedDict

from tools.frames.frame_extractor import FrameExtractor


class ClipsToFramesError(Exception):
    pass

class ClipsToFrames(object):
    def __init__(self, clip_path, annotation_path, frame_path):
        self._clip_path = clip_path
        self._annotation_path = annotation_path
        self._frame_path = frame_path

        self._clip_ids = []
        self._ensure_files_exit()

    def _ensure_files_exit(self):
        assert os.path.exists(self._annotation_path), \
                'Annotation path does not exist: {}'.format(self._annotation_path)
        assert os.path.exists(self._frame_path), \
                'Frame path does not exist: {}'.format(self._frame_path)
        self._clip_ids = [clipId for clipId in os.listdir(self._annotation_path)
                    if os.path.isdir(os.path.join(self._annotation_path, clipId))]
        for clipId in self._clip_ids:
            clipFile = os.path.join(self._clip_path, '{}.mp4'.format(clipId))
            assert os.path.exists(clipFile), \
                    'Clip mp4 does not exist for clipId: {}'.format(clipId)
            fnsFile = os.path.join(self._annotation_path, clipId, 'frame_numbers.txt')
            assert os.path.exists(fnsFile), \
                    'frame_numbers.txt does not exist for clipId: {}'.format(clipId)

    def extract(self):
        for clipId in self._clip_ids:
            clipFile = os.path.join(self._clip_path, '{}.mp4'.format(clipId))
            fnsFile = os.path.join(self._annotation_path, clipId, 'frame_numbers.txt')
            outBasePath = os.path.join(self._frame_path, clipId)
            frameExtractor = FrameExtractor(clipFile, outBasePath)
            frameExtractor.extract_based_on_file(fnsFile)
