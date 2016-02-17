import os
import pickle # TODO: replace with HDF5

from khajuri.multi.task import Task
from tools.files.file_utils import FileUtils


class FileSaverError(Exception):
    pass

class FileSaver(Task):
    def __init__(self):
        pass

    def start(self):
        pass

    def process(self, clip):
        folder = os.path.dirname(clip.result_path)
        FileUtils.mkdir_p(folder)
        # delete framedb prior to saving
        clip.framedb = None
        with open(clip.result_path, 'wb') as f:
            pickle.dump(clip, f)
        # delete predb prior to passing through rabbit queue
        # TODO: remove fc7 raw scores once pp is done
        clip.predb = None
        return clip

    def __str__(self):
        return 'FileSaver'
