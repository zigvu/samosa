import logging
import threading
import json

from khajuri.configs.khajuri_config import khajuri_cfg

from messaging.infra.rpc_client import RpcClient
from messaging.messages.clip_eval_details import ClipEvalDetails
from messaging.messages.header import Header

class ResultWriteClient(object):
    """Write result data to nimki"""

    def __init__(self):
        """Initialize values"""
        self.nimkiRpcClient = RpcClient(
            khajuri_cfg.RABBIT.ROUTES.AMQP_URL,
            khajuri_cfg.RABBIT.ROUTES.EXCHANGE,
            khajuri_cfg.RABBIT.ROUTES.RESULT_OUT_SERVER_ROUTING_KEY,
            khajuri_cfg.RABBIT.ROUTES.RESULT_OUT_SERVER_RESPONSE_ROUTING_KEY
        )
        self.header = Header.dataRequest()

    def write(self, clip):
        clipEvalMessage = clip.clip_eval_details
        # localResultPath will be inferred in nimki
        return self.nimkiRpcClient.call(self.header.message, json.dumps(clipEvalMessage.message))

    def writePoisonPill(self):
        clipEvalMessage = ClipEvalDetails()
        return self.nimkiRpcClient.call(self.header.message, json.dumps(clipEvalMessage.message))
