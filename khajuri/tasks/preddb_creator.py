from khajuri.multi.task import Task

from chia.test.frame_evaluater import FrameEvaluater

class PreddbCreatorError(Exception):
    pass

class PreddbCreator(Task):
    def __init__(self, gpuId):
        self.gpuId = gpuId

    def start(self):
        self.frameEvaluater = FrameEvaluater(self.gpuId)

    def process(self, clip):
        clip.preddb = self.frameEvaluater.evaluate(clip.framedb)
        # delete framedb after evaluation is done
        clip.framedb = None
        return clip

    def __str__(self):
        return 'PreddbCreator'
