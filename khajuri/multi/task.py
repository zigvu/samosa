class Task(object):
    def __init__(self, taskName):
        self.taskName = taskName

    def start(self):
        raise NotImplementedError

    def process(self, obj):
        raise NotImplementedError

    def __str__(self):
        return self.taskName
