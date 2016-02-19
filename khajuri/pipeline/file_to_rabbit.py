import logging
import threading

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

class FileToRabbit(threading.Thread):
    def __init__(self, fileStoredQueue):
        threading.Thread.__init__(self)
        self.fileStoredQueue = fileStoredQueue

    def run(self):
        logging.info('Starting FileToRabbit thread')
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            # TODO: setup rabbit
            pass
        while True:
            clip = self.fileStoredQueue.get()
            if clip is None:
                # Poison pill means shutdown
                logging.info('Exiting FileToRabbit thread')
                self.fileStoredQueue.task_done()
                break
            if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                # TODO: write to rabbit
                pass
            logging.debug('FileToRabbit: process clip: {}'.format(clip.clip_id))
            self.fileStoredQueue.task_done()
        return
