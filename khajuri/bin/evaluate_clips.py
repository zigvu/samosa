#!/usr/bin/env python

import logging
import numpy as np
import _init_paths

from khajuri.pipeline.run_pipeline import RunPipeline
from khajuri.multi.clip import Clip

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    runPipeline = RunPipeline()
    runPipeline.start()
    # BEGIN TODO: remove
    for i in xrange(10):
        nextClip = Clip()
        nextClip.clip_id = i
        nextClip.clip_path = 'filename'
        runPipeline.clipdbQueue.put(nextClip)
        logging.debug('Dummy putting clip id {} in queue'.format(nextClip.clip_id))
    # END TODO: remove
    runPipeline.join()
