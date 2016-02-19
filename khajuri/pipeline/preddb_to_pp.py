import os
import logging

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.worker import Worker

class PreddbToPpError(Exception):
    pass

class PreddbToPp(object):
    def __init__(self, preddbQueue, ppQueue):
        self.preddbQueue = preddbQueue
        self.ppQueue = ppQueue
        self.workers = []
        for i in xrange(khajuri_cfg.PIPELINE.PP_NUM_WORKERS):
            if khajuri_cfg.PIPELINE.IS_DRY_RUN:
                from khajuri.tasks.dummy_task import DummyTask
                postProcessor = DummyTask('PostProcessor')
            else:
                from khajuri.tasks.post_processor import PostProcessor
                postProcessor = PostProcessor(
                        khajuri_cfg.PIPELINE.PP_SCORE_THRESHS,
                        khajuri_cfg.PIPELINE.PP_INTER_THRESHS)
            worker = Worker(postProcessor, self.preddbQueue, self.ppQueue)
            self.workers.append(worker)
        self.numOfWorkers = len(self.workers)
        logging.info("Created {} PostProcessor workers".format(self.numOfWorkers))

    def start(self):
        for worker in self.workers:
            worker.start()

    def join(self):
        for worker in self.workers:
            worker.join()
