from   .SeparableConvFlowLayer import SeparableConvFlowLayer
import  torch
from    torch.nn import Module
class SeparableConvFlowModule(Module):
    def __init__(self, filtersize):
        super(SeparableConvFlowModule, self).__init__()
        self.filtersize = filtersize

    def forward(self, input1, input2, input3):
        return SeparableConvFlowLayer.apply(input1, input2, input3, self.filtersize)
