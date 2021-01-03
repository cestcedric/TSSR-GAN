import numpy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunAVGMeter(object):
    '''
    Computes and stores the running average
    '''
    def __init__(self, val = 0, size = 100):
        self.val = val
        self.size = size
        self.reset()

    def reset(self):
        self.storage = []
        self.filled = 0
        self.full = False
        self.compute_avg()

    def update(self, val, weight = 1):
        self.storage = [val*weight] + (self.storage[:-1] if self.full else self.storage)
        self.filled += 1
        self.full = self.filled >= self.size
        self.compute_avg()

    def compute_avg(self):
        self.avg = numpy.mean(self.storage if len(self.storage) > 0 else self.val)
