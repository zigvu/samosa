import os
import logging
from collections import OrderedDict
from multiprocessing import Process, Queue
import numpy as np

from fast_rcnn.config import cfg

# TODO: incomplete


class ImdbCreatorError(Exception):
    pass

class ImdbCreator(Process):
    def __init__(self, clipQueue, imdbQueue):
        super(ImdbCreator, self).__init__()
        self.clipQueue = clipQueue
        self.imdbQueue = imdbQueue

    def run(self):
        logging.info("Started test creator")
        while True:
            clip = self.clipQueue.get()
            if clip == None:
                break
