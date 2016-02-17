from easydict import EasyDict as edict

__CH = edict()
# Consumers can get config by:
#   from khajuri.configs.khajuri_config import khajuri_cfg
khajuri_cfg = __CH

# ------------------------------------------------------------------------------
# Rabbit
khajuri_cfg.RABBIT = edict()
khajuri_cfg.RABBIT.IS_RABBIT_RUN = False
# Pipeline
khajuri_cfg.PIPELINE = edict()
khajuri_cfg.PIPELINE.IS_DRY_RUN = True
khajuri_cfg.PIPELINE.CLIP_Q_SIZE = 10
khajuri_cfg.PIPELINE.FRAMEDB_Q_SIZE = 5
khajuri_cfg.PIPELINE.FRAMEDB_NUM_WORKERS = 2
khajuri_cfg.PIPELINE.PREDDB_Q_SIZE = 5
khajuri_cfg.PIPELINE.TEMPFS = '/tmp/pipeline'
khajuri_cfg.PIPELINE.EVALUATE_FPS = 5
khajuri_cfg.PIPELINE.GPU_IDS = [0]
