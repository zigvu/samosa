import logging
from time import sleep

from khajuri.multi.task import Task

class DummyTask(Task):
    def __init__(self, taskName):
        Task.__init__(self, taskName)

    def start(self):
        pass

    def process(self, clip):
        logging.debug('{}: process clip: {}'.format(self.taskName, clip.clip_id))
        sleep(1)
        return clip
