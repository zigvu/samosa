# Note: this class is serialized and passed through queues - so no methods
# are allowed as per pickle protocol

class Clip(object):
    def __init__(self):
        self.clip_id = None
        self.clip_path = None
        self.framedb = None
        self.predb = None
        self.result_path = None
