import numpy
import os
import torch
from   torch.utils.tensorboard import SummaryWriter
import utils.tools as tools

class Logger:

    def __init__(self, model_id, experiment_id):
        self.model_id = model_id
        self.writer = SummaryWriter(os.path.join('runs', experiment_id, model_id))
        self.offset = 0

    def log_scalars(self, tag, tag_value_dict, epoch, n_batch, num_batches):
        step = Logger._step(epoch, n_batch, num_batches, self.offset)
        self.writer.add_scalars(main_tag = tag, tag_scalar_dict = tag_value_dict, global_step = step)
        
    def log_histogram(self, tag, values, epoch, n_batch, num_batches):
        step = Logger._step(epoch, n_batch, num_batches, self.offset)
        values = numpy.array([ (v**2).sum().sqrt() if not v == None else 0 for v in values ])
        self.writer.add_histogram(tag = tag, values = values, global_step = step)
    
    def gradient_frames(self, tag, gradient, epoch, n_batch, num_batches):
        step = Logger._step(epoch, n_batch, num_batches, self.offset)
        self.writer.add_images(tag = tag, img_tensor = gradient, global_step = step)

    def save_images(self, tag, image, epoch, n_batch, num_batches):
        step = Logger._step(epoch, n_batch, num_batches, self.offset)
        self.writer.add_images(tag = tag, img_tensor = image, global_step = step)

    def save_model(self, model, images):
        self.writer.add_graph(model, images)

    def set_offset(self, offset):
        self.offset = offset

    @staticmethod
    # If offset > 0, epoch 0 is of variable length in included in offset
    def _step(epoch, n_batch, num_batches, offset = 0):
        return offset + n_batch + (epoch - (offset > 0)) * num_batches
        # return epoch * num_batches + n_batch + offset
