import logging
import threading

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

class RabbitToClip(threading.Thread):
    def __init__(self, clipdbQueue, exitQueue):
        threading.Thread.__init__(self)
        self.clipdbQueue = clipdbQueue
        self.exitQueue = exitQueue

    def run(self):
        logging.info('Starting RabbitToClip thread')
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            # TODO: setup rabbit
            pass
        while True:
            if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
                # TODO: receive from rabbitmq
                pass
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
        return
