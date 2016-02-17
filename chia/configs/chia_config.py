from easydict import EasyDict as edict

__CH = edict()
# Consumers can get config by:
#   from chia.configs.chia_config import chia_cfg
chia_cfg = __CH

# ------------------------------------------------------------------------------
# Train
chia_cfg.TRAIN = edict()
# Iteration
chia_cfg.TRAIN.ITERATION_ID = None
chia_cfg.TRAIN.PARENT_ITERATION_ID = None
chia_cfg.TRAIN.IS_MAJOR_ITERATION = False
chia_cfg.TRAIN.MAX_ITERS = 100
# Classes
chia_cfg.TRAIN.POSITIVE_CLASSES = []
chia_cfg.TRAIN.AVOID_CLASSES = []
# Folders and files
chia_cfg.TRAIN.FOLDERS = edict()
chia_cfg.TRAIN.FOLDERS.OUTPUT = '/tmp'
chia_cfg.TRAIN.FILES = edict()

# ------------------------------------------------------------------------------
# Test
chia_cfg.TEST = edict()
# Iteration
chia_cfg.TEST.ITERATION_ID = None
# Classes
chia_cfg.TEST.POSITIVE_CLASSES = []
chia_cfg.TEST.AVOID_CLASSES = []
# Folders and files
chia_cfg.TEST.FOLDERS = edict()
chia_cfg.TEST.FOLDERS.OUTPUT = '/tmp'
chia_cfg.TEST.FILES = edict()
# Test thresholds
chia_cfg.TEST.NMS = edict()
chia_cfg.TEST.NMS.DET_THRESH = 0.1
chia_cfg.TEST.NMS.BBOX_PER_IMG = 100
