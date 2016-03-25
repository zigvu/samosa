import logging
import threading
from multiprocessing import JoinableQueue

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

from messaging.connections.clip_ingest_server import ClipIngestServer

class RabbitToClip(threading.Thread):
    def __init__(self, clipdbQueue, exitQueue):
        threading.Thread.__init__(self)
        self.clipdbQueue = clipdbQueue
        self.exitQueue = exitQueue

    def run(self):
        logging.info('Starting RabbitToClip thread')
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            clipIngestQueue = JoinableQueue()
            clipIngestThread = ClipIngestServer(clipIngestQueue)
            clipIngestThread.start()
        while True:
            if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                clip = clipIngestQueue.get()
                clipIngestQueue.task_done()
            else:
                clip = Clip()
                clip.clip_id = None
            if clip.clip_id is None:
                # Poison pill means shutdown
                logging.info('Exiting RabbitToClip thread')
                # notify exit listeners
                self.exitQueue.put(None)
                break
            logging.debug('RabbitToClip: process clip: {}'.format(clip.clip_id))
            self.clipdbQueue.put(clip)
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            clipIngestQueue.join()
            clipIngestThread.join()
        return
