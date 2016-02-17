import logging
from time import sleep

from khajuri.multi.task import Task

class DummyTask(Task):
    def __init__(self):
        pass

    def start(self):
        pass

    def process(self, clip):
        logging.debug('Dummy process clip {}'.format(clip.clip_id))
        sleep(1)
        return clip

    def __str__(self):
        return 'DummyTask'
