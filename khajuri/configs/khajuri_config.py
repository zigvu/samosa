import socket
from easydict import EasyDict as edict

__CH = edict()
# Consumers can get config by:
#   from khajuri.configs.khajuri_config import khajuri_cfg
khajuri_cfg = __CH

# ------------------------------------------------------------------------------
# Rabbit
# ----------------
khajuri_cfg.RABBIT = edict()
khajuri_cfg.RABBIT.IS_RABBIT_RUN = True
khajuri_cfg.RABBIT.ROUTES = edict()
khajuri_cfg.RABBIT.ROUTES.AMQP_URL = 'localhost'
khajuri_cfg.RABBIT.ROUTES.EXCHANGE = 'samosa.khajuri_pipeline'
khajuri_cfg.RABBIT.ROUTES.CLIP_IN_SERVER_ROUTING_KEY = 'samosa.khajuri_pipeline.clip_in.server.{}'.format(socket.gethostname())
khajuri_cfg.RABBIT.ROUTES.RESULT_OUT_SERVER_ROUTING_KEY = 'samosa.khajuri_pipeline.result_out.server.{}'.format(socket.gethostname())
khajuri_cfg.RABBIT.ROUTES.RESULT_OUT_SERVER_RESPONSE_ROUTING_KEY = 'samosa.khajuri_pipeline.result_out.client.{}'.format(socket.gethostname())
# Pipeline
# ----------------
khajuri_cfg.PIPELINE = edict()
khajuri_cfg.PIPELINE.IS_DRY_RUN = False
khajuri_cfg.PIPELINE.TEMPFS = '/tmp/pipeline'
khajuri_cfg.PIPELINE.EVALUATE_FPS = 5
khajuri_cfg.PIPELINE.GPU_IDS = [0]
# queues and workers
khajuri_cfg.PIPELINE.FRAMEDB_Q_SIZE = 5
khajuri_cfg.PIPELINE.FRAMEDB_NUM_WORKERS = 2
khajuri_cfg.PIPELINE.PREDDB_Q_SIZE = 5
# post-process
khajuri_cfg.PIPELINE.PP_Q_SIZE = 5
khajuri_cfg.PIPELINE.PP_NUM_WORKERS = 2
khajuri_cfg.PIPELINE.PP_SCORE_THRESHS = [round(0.1 * i, 1) for i in xrange(1, 10)]
khajuri_cfg.PIPELINE.PP_INTER_THRESHS = [round(0.2 * i, 1) for i in xrange(1, 5)]
