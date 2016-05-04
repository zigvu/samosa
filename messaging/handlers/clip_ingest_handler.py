import logging
import json
import os

from khajuri.configs.khajuri_config import khajuri_cfg
from khajuri.multi.clip import Clip

from messaging.messages.clip_eval_details import ClipEvalDetails
from messaging.messages.header import Header

class ClipIngestHandler(object):
    """Handle clip data from nimki"""

    def __init__(self, clipIngestQueue):
        """Initialize values"""
        self.clipIngestQueue = clipIngestQueue
        self.outputPath = khajuri_cfg.PIPELINE.OUTPUT_BASE_FOLDER

    def handle(self, headers, message):
        logging.debug("Putting on queue, clip: {}".format(message['clipId']))

        clipEvalHeaders = Header(headers)
        clipEvalMessage = ClipEvalDetails(message)
        if khajuri_cfg.RABBIT.IS_RABBIT_RUN:
            clip = Clip()
            clip.clip_id = clipEvalMessage.message['clipId']
            if clip.clip_id == '':
                clip.clip_id = None
            clip.clip_path = clipEvalMessage.message['localClipPath']
            clip.clip_eval_details = clipEvalMessage
            clip.result_path = {
                'base_path': self.outputPath,
                'pickle': os.path.join(self.outputPath, '{}.pkl'.format(clip.clip_id)),
                'json': os.path.join(self.outputPath, '{}.json'.format(clip.clip_id))
            }
            self.clipIngestQueue.put(clip)
        else:
            self.clipIngestQueue.put(clipEvalMessage)

        clipEvalHeaders.setStateSuccess()
        responseHeaders = clipEvalHeaders.message
        responseMessage = clipEvalMessage.message
        return responseHeaders, json.dumps(responseMessage)

    def __call__(self, headers, message):
        return self.handle(headers, json.loads(message))
