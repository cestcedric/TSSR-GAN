# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import filterinterpolation_cuda as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class FilterInterpolationLayer(Function):
    def __init__(self):
        super(FilterInterpolationLayer,self).__init__()

    @staticmethod
    def forward(ctx, input1, input2, input3):

        assert(input1.is_contiguous())
        assert(input2.is_contiguous())
        assert(input3.is_contiguous())

        if input1.is_cuda:
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        else:
            output = torch.FloatTensor(input1.data.size())
            my_lib.FilterInterpolationLayer_cpu_forward(input1, input2, input3, output)

        ctx.save_for_backward(input1, input2, input3)
        # the function returns the output to its caller
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2, input3 = ctx.saved_tensors
        gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()
        gradinput3 = torch.cuda.FloatTensor().resize_(input3.size()).zero_()
        if input1.is_cuda:
            err = my_lib.FilterInterpolationLayer_gpu_backward(input1,input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        else:
            err = my_lib.FilterInterpolationLayer_cpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        return gradinput1, gradinput2, gradinput3

# calculate the weights of flow         
class WeightLayer(Function):
    def __init__(self):
        super(WeightLayer,self).__init__()

    # flow1_grad
    @staticmethod
    def forward(ctx, input1, input2, input3, lambda_e, lambda_v, Nw):
        output =  torch.zeros(input1.size(0), 1, input1.size(2), input1.size(3))

        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.WeightLayer_gpu_forward(input1, input2, input3, output, lambda_e, lambda_v, Nw)
            if err != 0 :
                print(err)
        else:
            err = my_lib.WeightLayer_cpu_forward(input1, input2, input3, output, lambda_e, lambda_v, Nw)
            if err != 0 :
                print(err)

        ctx.save_for_backward(input1.contiguous(), input2.contiguous(), input3.contiguous(), output)
        ctx.device = torch.cuda.current_device() if input1.is_cuda else -1
        ctx.lambda_e = lambda_e
        ctx.lambda_v = lambda_v
        ctx.Nw = Nw

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2, input3, output = ctx.saved_tensors

        gradinput1 = torch.zeros(input1.size())
        gradinput2 = torch.zeros(input2.size())
        gradinput3 = torch.zeros(input3.size())

        if input1.is_cuda:
            gradinput1 = gradinput1.cuda(ctx.device)
            gradinput2 = gradinput2.cuda(ctx.device)
            gradinput3 = gradinput3.cuda(ctx.device)
            err = my_lib.WeightLayer_gpu_backward(input1, input2, input3, output, gradoutput, gradinput1, gradinput2, gradinput3, ctx.lambda_e, ctx.lambda_v, ctx.Nw)
            if err != 0 :
                print(err)
        else:
            err = my_lib.WeightLayer_cpu_backward(input1, input2, input3, output, gradoutput, gradinput1, gradinput2, gradinput3, ctx.lambda_e, ctx.lambda_v, ctx.Nw)
            if err != 0 :
                print(err)

        return gradinput1, gradinput2, gradinput3, None, None, None
  
class PixelValueLayer(Function):
    def __init__(self):
        super(PixelValueLayer,self).__init__()

    @staticmethod
    def forward(ctx, input1, input3, flow_weights, sigma_d, tao_r, Prowindow):
        output = torch.zeros(input1.size())
        
        if input1.is_cuda:
            output = output.cuda()            
            err = my_lib.PixelValueLayer_gpu_forward(input1, input3, flow_weights, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelValueLayer_cpu_forward(input1, input3, flow_weights, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)

        ctx.save_for_backward(input1.contiguous(), input3.contiguous(), flow_weights.contiguous())
        ctx.device = torch.cuda.current_device() if input1.is_cuda else -1
        ctx.sigma_d = sigma_d
        ctx.tao_r = tao_r
        ctx.Prowindow = Prowindow

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input3, flow_weights = ctx.saved_tensors

        gradinput1 = torch.zeros(input1.size())
        gradinput3 = torch.zeros(input3.size())
        gradflow_weights = torch.zeros(flow_weights.size())

        if input1.is_cuda:
            gradinput1 = gradinput1.cuda(ctx.device)
            gradinput3 = gradinput3.cuda(ctx.device)
            gradflow_weights = gradflow_weights.cuda(ctx.device)
            err = my_lib.PixelValueLayer_gpu_backward(input1, input3, flow_weights, gradoutput, gradinput1, gradinput3, gradflow_weights, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelValueLayer_cpu_backward(input1, input3, flow_weights, gradoutput, gradinput1, gradinput3, gradflow_weights, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)
        return gradinput1, gradinput3, gradflow_weights, None, None, None

class PixelWeightLayer(Function):
    def __init__(self):
        super(PixelWeightLayer,self).__init__()

    @staticmethod
    def forward(ctx, input3, flow_weights, threshold, sigma_d, tao_r, Prowindow):
        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)])

        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.PixelWeightLayer_gpu_forward(input3, flow_weights, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelWeightLayer_cpu_forward(input3, flow_weights, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)

        ctx.save_for_backward(input3.contiguous(), flow_weights.contiguous(), output)
        ctx.device = torch.cuda.current_device() if input3.is_cuda else -1
        ctx.sigma_d = sigma_d
        ctx.tao_r = tao_r
        ctx.Prowindow = Prowindow
        ctx.threshold = threshold

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input3, flow_weights, output = ctx.saved_tensors

        gradinput3 = torch.zeros(input3.size())
        gradflow_weights = torch.zeros(flow_weights.size())

        if input3.is_cuda:
            gradinput3 = gradinput3.cuda(ctx.device)
            gradflow_weights = gradflow_weights.cuda(ctx.device)
            err = my_lib.PixelWeightLayer_gpu_backward(input3, flow_weights, output, gradoutput, gradinput3, gradflow_weights, ctx.threshold, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.PixelWeightLayer_cpu_backward(input3, flow_weights, output, gradoutput, gradinput3, gradflow_weights, ctx.threshold, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)

        return gradinput3, gradflow_weights, None, None, None, None
		
class ReliableWeightLayer(Function):
    def __init__(self):
        super(ReliableWeightLayer,self).__init__()

    @staticmethod
    def forward(ctx, input3, threshold, sigma_d, tao_r, Prowindow):
        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)] )

        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.ReliableWeightLayer_gpu_forward(input3, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.ReliableWeightLayer_cpu_forward(input3, output, sigma_d, tao_r, Prowindow)
            if err != 0 :
                print(err)

        ctx.save_for_backward(input3.contiguous(), output)
        ctx.device = torch.cuda.current_device() if input3.is_cuda else -1
        ctx.sigma_d = sigma_d
        ctx.tao_r = tao_r
        ctx.Prowindow = Prowindow
        ctx.threshold = threshold

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input3, output = ctx.saved_tensors

        gradinput3 = torch.zeros(input3.size())
        
        if input3.is_cuda:
            gradinput3 = gradinput3.cuda(ctx.device)
            err = my_lib.ReliableWeightLayer_gpu_backward(input3, output, gradoutput, gradinput3, ctx.threshold, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)
        else:
            err = my_lib.ReliableWeightLayer_cpu_backward(input3, output, gradoutput, gradinput3, ctx.threshold, ctx.sigma_d, ctx.tao_r, ctx.Prowindow)
            if err != 0 :
                print(err)

        return gradinput3, None, None, None, None