# modules/AdaptiveInterpolationLayer.py
from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from .FilterInterpolationLayer import FilterInterpolationLayer,WeightLayer, PixelValueLayer,PixelWeightLayer,ReliableWeightLayer

class FilterInterpolationModule(Module):
    def __init__(self):
        super(FilterInterpolationModule, self).__init__()
        # self.f = FilterInterpolationLayer()

    def forward(self, input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)

class AdaptiveWeightInterpolationModule(Module):
    def __init__(self,  
        training = False,
        threshold = 1e-6,
        lambda_e = 30.0/255.0,
        lambda_v = 1.0,
        Nw = 3.0,
        sigma_d = 1.5,
        tao_r = 0.05,
        Prowindow = 2):
        super(AdaptiveWeightInterpolationModule, self).__init__()

        self.padder1 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.padder2 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.sigma_d = sigma_d
        self.tao_r = tao_r
        self.Prowindow = Prowindow
        self.threshold = threshold
        self.lambda_e = lambda_e
        self.lambda_v = lambda_v
        self.Nw = Nw
        self.pixelvalue_params = {'sigma_d': sigma_d, 'tao_r': tao_r, 'Prowindow': Prowindow}
        self.pixelweight_params = {'threshold': 101*threshold, 'sigma_d': sigma_d, 'tao_r': tao_r, 'Prowindow': Prowindow}
        self.reliableweight_params = {'threshold': 101*threshold, 'sigma_d': sigma_d, 'tao_r': tao_r, 'Prowindow': Prowindow}
        self.weight_params = {'lambda_e': lambda_e, 'lambda_v': lambda_v, 'Nw': Nw}
        
        self.training = training
        self.threshold = threshold
        return

    # input1 ==> ref1 image
    # input2 ==> ref2 image
    # input3 ==> ref1 flow
    # input4 ==> ref2 flow
    def forward(self, input1, input2, input3, input4):
        flow_weight1 = WeightLayer.apply(input1, input2, input3, self.lambda_e, self.lambda_v, self.Nw)
        p1 = PixelValueLayer.apply(input1, input3, flow_weight1, self.sigma_d, self.tao_r, self.Prowindow)
        pw1 = PixelWeightLayer.apply(input3, flow_weight1, 101*self.threshold, self.sigma_d, self.tao_r, self.Prowindow)
        p1_r, p1_g, p1_b = torch.split(p1, 1, dim=1)
        i1_r = (p1_r) / (pw1+self.threshold)
        i1_g = (p1_g) / (pw1+self.threshold)
        i1_b = (p1_b) / (pw1+self.threshold)
        r1 = pw1
        rw1 = ReliableWeightLayer.apply(input3, 101*self.threshold, self.sigma_d, self.tao_r, self.Prowindow)
        w1 = (r1) / (rw1+self.threshold)

        flow_weight2 = WeightLayer.apply(input2, input1, input4, self.lambda_e, self.lambda_v, self.Nw)
        p2 = PixelValueLayer.apply(input2, input4, flow_weight2, self.sigma_d, self.tao_r, self.Prowindow)
        pw2 = PixelWeightLayer.apply(input4, flow_weight2, 101*self.threshold, self.sigma_d, self.tao_r, self.Prowindow)
        p2_r, p2_g, p2_b = torch.split(p2, 1, dim=1)
        i2_r = (p2_r) / (pw2+self.threshold)
        i2_g = (p2_g) / (pw2+self.threshold)
        i2_b = (p2_b) / (pw2+self.threshold)
        r2 = pw2
        rw2 = ReliableWeightLayer.apply(input4, 101*self.threshold, self.sigma_d, self.tao_r, self.Prowindow)
        w2 = (r2) / (rw2+self.threshold)

        w = w1+w2
        i_r = (i1_r * w1 + i2_r * w2) / (w + self.threshold) #(w1 + w2)
        i_g = (i1_g * w1 + i2_g * w2) / (w + self.threshold) #(w1 + w2)
        i_b = (i1_b * w1 + i2_b * w2) / (w + self.threshold) #(w1 + w2)
        if not self.training:
            i_r[w <= 10*self.threshold] = 0
            i_g[w <= 10*self.threshold] = 0
            i_b[w <= 10*self.threshold] = 0
            w[w <= 10 *self.threshold] = 0
        i = torch.cat((i_r, i_g, i_b), dim=1)
        return i
