import logging

from khajuri.multi.task import Task
from chia.datasets.test_creator import TestCreator

class FramedbCreatorError(Exception):
    pass

class FramedbCreator(Task):
    def __init__(self, fps, temp_path):
        Task.__init__(self, 'FramedbCreator')
        self.testCreator = TestCreator(fps, temp_path)

    def start(self):
        pass

    def process(self, clip):
        logging.debug('{}: process clip: {}'.format(self.taskName, clip.clip_id))
        clip.framedb = self.testCreator.get_framedb(clip.clip_path)
        return clip
