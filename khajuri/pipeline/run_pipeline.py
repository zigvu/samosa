from multiprocessing import JoinableQueue

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.pipeline.rabbit_to_clip import RabbitToClip
from khajuri.pipeline.clip_to_framedb import ClipToFramedb
from khajuri.pipeline.framedb_to_preddb import FramedbToPreddb
from khajuri.pipeline.file_to_rabbit import FileToRabbit

class RunPipelineError(Exception):
    pass

class RunPipeline(object):
    def __init__(self):
        self.exitQueue = JoinableQueue()
        self.clipdbQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.CLIP_Q_SIZE)
        self.framedbQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.FRAMEDB_Q_SIZE)
        self.preddbQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.PREDDB_Q_SIZE)

    def start(self):
        # setup pipeline
        self.rabbitToClip = RabbitToClip(self.clipdbQueue, self.exitQueue)
        self.clipToFramedb = ClipToFramedb(self.clipdbQueue, self.framedbQueue)
        self.framedbToPreddb = FramedbToPreddb(self.framedbQueue, self.preddbQueue)
        self.fileToRabbit = FileToRabbit(self.preddbQueue)
        # start threads after spawning processes
        self.clipToFramedb.start()
        self.framedbToPreddb.start()
        self.rabbitToClip.start()
        self.fileToRabbit.start()

    def join(self):
        # wait to join until rabbit tells to do so
        while True:
            if self.exitQueue.get() is None:
                self.exitQueue.task_done()
                break
        self.exitQueue.join()
        # join queues
        for i in xrange(self.clipToFramedb.numOfWorkers):
            self.clipdbQueue.put(None)
        self.clipdbQueue.join()
        for i in xrange(self.framedbToPreddb.numOfWorkers):
            self.framedbQueue.put(None)
        self.framedbQueue.join()
        self.preddbQueue.put(None)
        self.preddbQueue.join()
        # join processes
        self.clipToFramedb.join()
        self.framedbToPreddb.join()
        # join threads
        self.rabbitToClip.join()
        self.fileToRabbit.join()
