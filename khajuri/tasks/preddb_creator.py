from khajuri.multi.task import Task
from khajuri.tasks.file_saver import FileSaver # TODO: move to after pp

from chia.test.frame_evaluater import FrameEvaluater

class PreddbCreatorError(Exception):
    pass

class PreddbCreator(Task):
    def __init__(self, gpuId):
        self.frameEvaluater = FrameEvaluater(gpuId)
        self.fileSaver = FileSaver()

    def process(self, clip):
        clip.preddb = self.frameEvaluater.evaluate(clipData)
        clip = self.fileSaver.process(clip) # TODO: move to after pp
        return clip

    def __str__(self):
        return 'PreddbCreator'
