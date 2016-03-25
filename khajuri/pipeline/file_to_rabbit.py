import logging
import threading

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

from messaging.connections.result_write_client import ResultWriteClient

class FileToRabbit(threading.Thread):
    def __init__(self, fileStoredQueue):
        threading.Thread.__init__(self)
        self.fileStoredQueue = fileStoredQueue

    def run(self):
        logging.info('Starting FileToRabbit thread')
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            resultWriteClient = ResultWriteClient()
        while True:
            clip = self.fileStoredQueue.get()
            if clip is None:
                # Poison pill means shutdown
                logging.info('Exiting FileToRabbit thread')
                if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                    resultWriteClient.writePoisonPill()
                self.fileStoredQueue.task_done()
                break
            if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                resultWriteClient.write(clip)
            logging.debug('FileToRabbit: process clip: {}'.format(clip.clip_id))
            self.fileStoredQueue.task_done()
        return
