class Task(object):
    def __init__(self):
        pass

    def start(self):
        raise NotImplementedError

    def process(self, obj):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
