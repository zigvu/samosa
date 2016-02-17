import logging
import threading

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

class FileToRabbit(threading.Thread):
    def __init__(self, preddbQueue):
        threading.Thread.__init__(self)
        self.preddbQueue = preddbQueue

    def run(self):
        logging.debug('Starting FileToRabbit thread')
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            # TODO: setup rabbit
            pass
        while True:
            clip = self.preddbQueue.get()
            if clip is None:
                # Poison pill means shutdown
                logging.debug('Exiting FileToRabbit thread')
                self.preddbQueue.task_done()
                break
            if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                # TODO: write to rabbit
                pass
            logging.debug('Done pred for clip id {}'.format(clip.clip_id))
            self.preddbQueue.task_done()
        return
