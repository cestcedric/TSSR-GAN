import torch
from   torchvision import models
from   utils import tools
# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/models/pretrained_networks.py
# Extracted values from 'vgg_19/conv2/conv2_2', 'vgg_19/conv3/conv3_4', 'vgg_19/conv4/conv4_4', 'vgg_19/conv5/conv5_4'

class vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.N_slices = 4
        for x in range(9):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 35):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    '''
    Assume input frame values in [0,1]
    '''
    def forward(self, x, normalize = True):
        x = x*2 - 1 
        x = self.slice1(x)
        conv2_2 = tools.normalize_tensor(x) if normalize else x
        x = self.slice2(x)
        conv3_4 = tools.normalize_tensor(x) if normalize else x
        x = self.slice3(x)
        conv4_4 = tools.normalize_tensor(x) if normalize else x
        x = self.slice4(x)
        conv5_4 = tools.normalize_tensor(x) if normalize else x
        return (conv2_2, conv3_4, conv4_4, conv5_4)
