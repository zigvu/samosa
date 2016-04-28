import logging

from khajuri.multi.task import Task
from khajuri.postprocess.find_confusions import FindConfusions
from khajuri.tasks.file_saver import FileSaver

class PostProcessorError(Exception):
    pass

class PostProcessor(Task):
    def __init__(self, scoreThreshs, interThreshs):
        Task.__init__(self, 'PostProcessor')
        self.findConfusions = FindConfusions(scoreThreshs, interThreshs)
        self.fileSaver = FileSaver()

    def start(self):
        pass

    def process(self, clip):
        logging.debug('{}: process clip: {}'.format(self.taskName, clip.clip_id))
        clip.confdb = self.findConfusions.confusions(clip.preddb)
        clip = self.fileSaver.process(clip)
        # after save, data not needed
        clip.predb = None
        clip.confdb = None
        return clip
