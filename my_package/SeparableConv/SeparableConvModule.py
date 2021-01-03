from functions.SeparableConvLayer import SeparableConvLayer
from torch.nn import Module

class SeparableConvModule(Module):
    def __init__(self, filtersize):
        super(SeparableConvModule, self).__init__()
        self.filtersize = filtersize
    def forward(self, input1, input2, input3):
        return SeparableConvLayer.apply(input1, input2, input3, self.filtersize)
