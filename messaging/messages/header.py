# NOTE: this file corresponds to messaging/messages/header.rb

class Header(object):
    """Header for communicating with Nimki"""

    def __init__(self, message=None):
        # type = "ping", "status", "data"
        # state = "request", "unknown", "success", "failure"
        self.message = message
        if self.message == None:
            self.message = {
                "type": "ping",
                "state": "success"
            }

    def setStateSuccess(self):
        self.message["state"] = "success"

    def isStateSuccess(self):
        return (self.message["state"] == "success")

    @staticmethod
    def dataRequest():
        return Header({"type": "data", "state": "request"})
