from khajuri.multi.task import Task
from chia.datasets.test_creator import TestCreator

class FramedbCreatorError(Exception):
    pass

class FramedbCreator(Task):
    def __init__(self, fps, temp_path):
        self.testCreator = TestCreator(fps, temp_path)

    def start(self):
        pass

    def process(self, clip):
        clip.framedb = self.testCreator.get_framedb(clip.clip_path)
        return clip

    def __str__(self):
        return 'FramedbCreator'
