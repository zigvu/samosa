from khajuri.multi.task import Task
from khajuri.postprocess.find_confusions import FindConfusions
from khajuri.tasks.file_saver import FileSaver

class PostProcessorError(Exception):
    pass

class PostProcessor(Task):
    def __init__(self, scoreThreshs, interThreshs):
        self.findConfusions = FindConfusions(scoreThreshs, interThreshs)

    def start(self):
        pass

    def process(self, clip):
        clip.confdb = self.findConfusions.confusions(clip.preddb)
        clip = self.fileSaver.process(clip)
        # after save, data not needed
        clip.predb = None
        clip.confdb = None
        return clip

    def __str__(self):
        return 'PostProcessor'
