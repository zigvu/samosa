import logging
import threading

from khajuri.configs.khajuri_config import khajuri_cfg

from messaging.handlers.clip_ingest_handler import ClipIngestHandler
from messaging.infra.rpc_server import RpcServer

class ClipIngestServer(threading.Thread):
    """Get clip data from nimki"""

    def __init__(self, clipIngestQueue):
        """Initialize values"""
        threading.Thread.__init__(self)
        self.clipIngestQueue = clipIngestQueue

    def run(self):
        logging.info('Starting ClipIngestServer thread')
        clipIngestHandler = ClipIngestHandler(self.clipIngestQueue)
        rpc = RpcServer((
            khajuri_cfg.RABBIT.ROUTES.AMQP_URL,
            khajuri_cfg.RABBIT.ROUTES.EXCHANGE,
            khajuri_cfg.RABBIT.ROUTES.CLIP_IN_SERVER_ROUTING_KEY,
            self.clipIngestQueue,)
        )
