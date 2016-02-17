import os
import pickle # TODO: replace with HDF5

from khajuri.multi.task import Task
from tools.files.file_utils import FileUtils


class FileSaverError(Exception):
    pass

class FileSaver(Task):
    def __init__(self):
        pass

    def process(self, clip):
        folder = os.path.dirname(clip.result_path)
        FileUtils.mkdir_p(folder)
        with open(clip.result_path, 'wb') as f:
            pickle.dump(clip, f)
        # delete large data structures prior to passing through queue again
        clip.framedb = None
        clip.predb = None
        return clip

    def __str__(self):
        return 'FileSaver'
