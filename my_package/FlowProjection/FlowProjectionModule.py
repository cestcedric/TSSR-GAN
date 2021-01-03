# modules/FlowProjectionModule.py
from torch.nn import Module
from .FlowProjectionLayer import FlowProjectionLayer

class FlowProjectionModule(Module):
    def __init__(self, requires_grad = True):
        super(FlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, input1):
        return FlowProjectionLayer.apply(input1, self.requires_grad)
