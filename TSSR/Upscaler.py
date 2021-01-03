import numpy
import torch
from   torch import nn
import torch.nn.functional as F
from   submodules.Upscaling.Blocks import EasyConv2d, RB
from   submodules.Upscaling.Math import nd_meshgrid
from   submodules.Upscaling.Scale import SpaceToDepth, Upsample
from   submodules.Upscaling.Utility import pad_if_divide, upsample
from   utils import tools

class Generator(nn.Module):
    """Generator in TecoGAN.

    Note: the flow estimation net `Fnet` shares with FRVSR.

    Args:
        filters: basic filter numbers [default: 64]
        num_rb: number of residual blocks [default: 16]
    """

    def __init__(self, channel, scale, filters, num_rb):
        super(Generator, self).__init__()
        rbs = []
        for i in range(num_rb):
            rbs.append(RB(filters, filters, 3, 'relu'))
        self.body = nn.Sequential(
                EasyConv2d(channel * (1 + scale ** 2), filters, 3, activation='relu'),
                *rbs,
                Upsample(filters, scale, 'deconv', activation='relu'),
                EasyConv2d(filters, channel, 3))

    def forward(self, x, prev, residual=None):
        """`residual` is the bicubically upsampled HR images"""
        sr = self.body(torch.cat((x, prev), dim=1))
        if residual is not None:
            sr += residual
        return sr


class Composer(nn.Module):
    def __init__(self, scale, channel, gain = 24, filters = 64, n_rb = 16, weights = None):
        super(Composer, self).__init__()
        self.gnet = Generator(channel = channel, scale = scale, filters = filters, num_rb = n_rb)
        self.warpper = STN(padding_mode='border')
        self.spd = SpaceToDepth(scale)
        self.scale = scale
        self.gain = gain

    def forward(self, lr, sr_pre, sr_warp, detach_fnet = False, preupsampled = None):
        """
        Args:
             lr: t_1 lr frame
             sr_pre: t_0 sr frame
             detach_fnet: detach BP to warping, flow estimation
        """
        bi = upsample(lr, self.scale) if preupsampled == None else preupsampled
        if detach_fnet:
            return self.gnet(lr, self.spd(sr_warp.detach()), bi)
        return self.gnet(lr, self.spd(sr_warp), bi)

class STN(nn.Module):
    """Spatial transformer network.
        For optical flow based frame warping.

    Args:
        mode: sampling interpolation mode of `grid_sample`
        padding_mode: can be `zeros` | `borders`
        normalized: flow value is normalized to [-1, 1] or absolute value
    """
    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize

    def forward(self, inputs, u, v=None, gain=1):
        batch = inputs.size(0)
        device = inputs.device
        mesh = nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
        mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
        # add flow to mesh
        if v is None:
            assert u.shape[1] == 2, "optical flow must have 2 channels"
            _u, _v = u[:, 0], u[:, 1]
        else:
            _u, _v = u, v

        if not self.norm:
        # flow needs to normalize to [-1, 1]
            h, w = inputs.shape[-2:]
            _u = _u / w * 2
            _v = _v / h * 2
        flow = torch.stack([_u, _v], dim=-1) * gain
        assert flow.shape == mesh.shape, \
            f"Shape mis-match: {flow.shape} != {mesh.shape}"
        mesh = mesh + flow
        return F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode)
