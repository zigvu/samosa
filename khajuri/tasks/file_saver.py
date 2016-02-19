import os
import pickle # TODO: replace with HDF5

from khajuri.multi.task import Task
from tools.files.file_utils import FileUtils


class FileSaverError(Exception):
    pass

class FileSaver(Task):
    def __init__(self):
        Task.__init__(self, 'FileSaver')

    def start(self):
        pass

    def process(self, clip):
        logging.debug('{}: process clip: {}'.format(self.taskName, clip.clip_id))
        folder = os.path.dirname(clip.result_path)
        FileUtils.mkdir_p(folder)
        # TODO: write to hdf5 instead of np
        with open(clip.result_path, 'wb') as f:
            pickle.dump(clip, f)
        return clip
