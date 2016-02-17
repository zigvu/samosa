import logging
import os
import multiprocessing
import threading

class ThreadWorker(threading.Thread):
    def __init__(self, task, input_queue, output_queue):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.task = task

    def run(self):
        logging.info('%s: Starting' % self)
        while True:
            next_object = self.input_queue.get()
            if next_object is None:
                # Poison pill means shutdown
                logging.info('%s: Exiting' % self)
                self.input_queue.task_done()
                # do not propagate None to output_queue: assume that
                # queue managers will take care of that
                break
            answer = self.task.process(next_object)
            self.input_queue.task_done()
            self.output_queue.put(answer)
        return

    def __str__(self):
        return '( %s )' % self.task


class ProcessWorker(multiprocessing.Process):
    def __init__(self, task, input_queue, output_queue):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.task = task
        self.task_done = False

    def run(self):
        logging.info('Starting Process %s with pid %s task as %s' %
                (self.name, self.pid, self.task))
        while True:
            next_object = self.input_queue.get()
            if next_object is None:
                # Poison pill means shutdown
                logging.info('%s: Exiting from Worker' % self)
                self.input_queue.task_done()
                # do not propagate None to output_queue: assume that
                # queue managers will take care of that
                break
            answer = self.task.process(next_object)
            self.input_queue.task_done()
            self.output_queue.put(answer)
        self.task_done = True
        return

    def __str__(self):
        return '( %s, %s )' % (self.task, self.pid)


class Worker(ProcessWorker):
    pass
