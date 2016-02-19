import os
import logging

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.worker import Worker

class FramedbToPreddbError(Exception):
    pass

class FramedbToPreddb(object):
    def __init__(self, framedbQueue, preddbQueue):
        self.framedbQueue = framedbQueue
        self.preddbQueue = preddbQueue
        self.workers = []
        for gpuId in khajuri_cfg.PIPELINE.GPU_IDS:
            if khajuri_cfg.PIPELINE.IS_DRY_RUN:
                from khajuri.tasks.dummy_task import DummyTask
                preddbCreator = DummyTask('PreddbCreator')
            else:
                from khajuri.tasks.preddb_creator import PreddbCreator
                preddbCreator = PreddbCreator(gpuId)
            worker = Worker(preddbCreator, self.framedbQueue, self.preddbQueue)
            self.workers.append(worker)
        self.numOfWorkers = len(self.workers)
        logging.info("Created {} PreddbCreator workers".format(self.numOfWorkers))

    def start(self):
        for worker in self.workers:
            worker.start()

    def join(self):
        for worker in self.workers:
            worker.join()
