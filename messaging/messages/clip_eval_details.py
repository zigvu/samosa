# NOTE: this file corresponds to messaging/messages/samosa/cap_eval_details.rb

class ClipEvalDetails(object):
    """Clip details for khajuri processing"""

    def __init__(self, message=None):
        self.category = 'samosa'
        self.name = 'clip_eval_details'
        self.message = message
        if self.message == None:
            self.message = {}
            self.message['category'] = self.category
            self.message['name'] = self.name
            self.message['trace'] = None
            self.message['capEvalId'] = None
            self.message['clipId'] = None
            self.message['localClipPath'] = None
            self.message['storageHostname'] = None
            self.message['storageClipPath'] = None
            self.message['localResultPath'] = None
            self.message['storageResultPath'] = None
            self.message['state'] = None

    def isSameType(self):
        return (
            (self.category == self.message['category']) and
            (self.name == self.message['name'])
        )
