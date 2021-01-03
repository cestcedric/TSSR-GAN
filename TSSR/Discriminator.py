from  .Blocks import Basic
import torch
import torch.nn as nn
from   utils import tools

class Discriminator(nn.Module):
    def __init__(self, in_channels = 0, training = False):

        # base class initialization
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.training = training            

        # Discriminator block layers
        self.block0 = Basic(in_channels = in_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.block1 = Basic(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.block2 = Basic(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.block3 = Basic(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.block4 = Basic(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)

        # Dense layer
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
            
        tools.initialize_weights(self)
            
        return 

    def forward(self, x):
        x  = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x  = self.dense(x4)
        if self.training:
            return x, [x1, x2, x3, x4]
        return x