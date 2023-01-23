# -*- coding: utf-8 -*-


# ToDo
# Borrowed from Carl's BAR project. Will need to make it into the base repo
class LossHistory(object):
    """History of the loss during training. (Lighter version of MetricHistory)

    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self):
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0

    def update(self, value):
        self.history.append(value)
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self, write=True):
        if write:
            self.epochs.append(self._avg)
            self.num_epochs += 1
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
