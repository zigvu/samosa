import os
import logging
from collections import OrderedDict
from multiprocessing import Process, Queue
import numpy as np

from khajuri.datasets.test_creator import TestCreator

from fast_rcnn.config import cfg

# TODO: incomplete

class FramedbCreatorError(Exception):
    pass

class FramedbCreator(Process):
    def __init__(self, clipdbQueue, framedbQueue):
        super(FramedbCreator, self).__init__()
        self.clipdbQueue = clipdbQueue
        self.framedbQueue = framedbQueue

    def run(self):
        logging.info("Started test creator")
        while True:
            clip = self.clipdbQueue.get()
            if clip == None:
                break
            testCreator = TestCreator(args.clip_path, 5, framedbPath)
