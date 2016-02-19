from multiprocessing import JoinableQueue

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.pipeline.rabbit_to_clip import RabbitToClip
from khajuri.pipeline.clip_to_framedb import ClipToFramedb
from khajuri.pipeline.framedb_to_preddb import FramedbToPreddb
from khajuri.pipeline.preddb_to_pp import PreddbToPp
from khajuri.pipeline.file_to_rabbit import FileToRabbit

class RunPipelineError(Exception):
    pass

class RunPipeline(object):
    def __init__(self):
        self.exitQueue = JoinableQueue()
        self.clipdbQueue = JoinableQueue()
        self.framedbQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.FRAMEDB_Q_SIZE)
        self.preddbQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.PREDDB_Q_SIZE)
        self.ppQueue = JoinableQueue(maxsize=khajuri_cfg.PIPELINE.PP_Q_SIZE)

    def start(self):
        # setup pipeline
        self.rabbitToClip = RabbitToClip(self.clipdbQueue, self.exitQueue)
        self.clipToFramedb = ClipToFramedb(self.clipdbQueue, self.framedbQueue)
        self.framedbToPreddb = FramedbToPreddb(self.framedbQueue, self.preddbQueue)
        self.preddbToPp = PreddbToPp(self.preddbQueue, self.ppQueue)
        self.fileToRabbit = FileToRabbit(self.ppQueue)
        # spawn processes
        self.clipToFramedb.start()
        self.framedbToPreddb.start()
        self.preddbToPp.start()
        # start threads after spawning processes
        self.rabbitToClip.start()
        self.fileToRabbit.start()

    def join(self):
        # wait to join until rabbit tells to do so
        while True:
            if self.exitQueue.get() is None:
                self.exitQueue.task_done()
                break
            else:
                logging.error("Unknown message in exitQueue")
                self.exitQueue.task_done()
        self.exitQueue.join()
        # join queues
        for i in xrange(self.clipToFramedb.numOfWorkers):
            self.clipdbQueue.put(None)
        self.clipdbQueue.join()
        for i in xrange(self.framedbToPreddb.numOfWorkers):
            self.framedbQueue.put(None)
        self.framedbQueue.join()
        for i in xrange(self.preddbToPp.numOfWorkers):
            self.preddbQueue.put(None)
        self.preddbQueue.join()
        self.ppQueue.put(None)
        self.ppQueue.join()
        # join processes
        self.clipToFramedb.join()
        self.framedbToPreddb.join()
        self.preddbToPp.join()
        # join threads
        self.rabbitToClip.join()
        self.fileToRabbit.join()
