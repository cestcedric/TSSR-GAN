import torch
import torch.nn as nn
from   utils import tools

class Basic(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding
        ):

        super(Basic, self).__init__()

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.relu = nn.LeakyReLU(
            negative_slope = 0.2,
            inplace = False
        )

        tools.initialize_weights(self)
            
        return

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, 
        in_channels,
        mid_channels,
        out_channels, 
        kernel_size, 
        stride, 
        padding
        ):

        super(Residual, self).__init__()

        self.scaleRes = not in_channels == out_channels

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = mid_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.conv2 = nn.Conv2d(
            in_channels = mid_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.relu = nn.LeakyReLU(
            negative_slope = 0.2,
            inplace = False
        )
        self.scale = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        tools.initialize_weights(self)
            
        return

    def forward(self, x):
        residual = self.scale(x) if self.scaleRes else x 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x


class Stub(nn.Module):
    def __init__(self, shape):
        super(Stub, self).__init__()
        self.shape = shape
        return

    def forward(self, x):
        return x.new_ones(self.shape)