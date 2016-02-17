import os
import logging

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.worker import Worker

class ClipToFramedbError(Exception):
    pass

class ClipToFramedb(object):
    def __init__(self, clipdbQueue, framedbQueue):
        self.clipdbQueue = clipdbQueue
        self.framedbQueue = framedbQueue
        self.workers = []
        for i in xrange(khajuri_cfg.PIPELINE.FRAMEDB_NUM_WORKERS):
            if khajuri_cfg.PIPELINE.IS_DRY_RUN:
                from khajuri.tasks.dummy_task import DummyTask
                framedbCreator = DummyTask()
            else:
                from khajuri.tasks.framedb_creator import FramedbCreator
                tempPath = os.path.join(khajuri_cfg.PIPELINE.TEMPFS, "clipToFramedb_{}".format(i))
                framedbCreator = FramedbCreator(khajuri_cfg.PIPELINE.EVALUATE_FPS, tempPath)
            worker = Worker(framedbCreator, self.clipdbQueue, self.framedbQueue)
            self.workers.append(worker)
        self.numOfWorkers = len(self.workers)
        logging.info("Created {} FramedbCreator workers".format(self.numOfWorkers))

    def start(self):
        for worker in self.workers:
            worker.start()

    def join(self):
        for worker in self.workers:
            worker.join()
