import json

from messaging.messages.clip_eval_details import ClipEvalDetails
from messaging.messages.header import Header

class ClipIngestHandler(object):
    """Handle clip data from nimki"""

    def __init__(self, clipdbQueue):
        """Initialize values"""
        self.clipdbQueue = clipdbQueue

    def handle(self, headers, message):
        print "Putting on queue, clip: {}".format(message['clipId'])

        clipEvalHeaders = Header(headers)
        clipEvalMessage = ClipEvalDetails(message)
        self.clipdbQueue.put(clipEvalMessage)

        clipEvalHeaders.setStateSuccess()
        responseHeaders = clipEvalHeaders.message
        responseMessage = clipEvalMessage.message
        return responseHeaders, json.dumps(responseMessage)

    def __call__(self, headers, message):
        return self.handle(headers, json.loads(message))
