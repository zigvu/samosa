"""Clobber edict values based on yaml file."""

import yaml
import numpy as np
from easydict import EasyDict as edict

class ClobberEdict(object):
    def __init__(self, initDict):
        self.initDict = initDict

    def clobber(self, filename):
        with open(filename, 'r') as f:
            yaml_cfg = edict(yaml.load(f))

        self._merge_a_into_b(yaml_cfg, self.initDict)
        return self.initDict

    def _merge_a_into_b(self, a, b):
        """Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a.
        """
        if type(a) is not edict:
            return

        for k, v in a.iteritems():
            # a must specify keys that are in b
            if not b.has_key(k):
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                    'for config key: {}').format(type(b[k]),
                                                                type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    self._merge_a_into_b(self, a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                b[k] = v
