# this is for wrapping the customized layer
import flowprojection_cuda as my_lib
import torch
from   torch.autograd import Function

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class FlowProjectionLayer(Function):
    def __init__(self,requires_grad):
        super(FlowProjectionLayer,self).__init__()
        self.requires_grad = requires_grad

    @staticmethod
    def forward(ctx, input1, requires_grad):
        assert(input1.is_contiguous())
        fillhole = 1 if requires_grad == False else 0

        if input1.is_cuda :
            count = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_gpu_forward(input1, count, output, fillhole)
        else:
            output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.FlowProjectionLayer_cpu_forward(input1, count, output, fillhole)
        if err != 0:
            print(err)

        ctx.save_for_backward(input1, count)
        ctx.fillhole = fillhole

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, count = ctx.saved_tensors

        if input1.is_cuda:
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_gpu_backward(input1, count, gradoutput, gradinput1)
            if err != 0 :
                print(err)
        else:
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_cpu_backward(input1, count, gradoutput, gradinput1)
            if err != 0:
                print(err)

        return gradinput1, None

class FlowFillholelayer(Function):
    def __init__(self):
        super(FlowFillholelayer,self).__init__()

    @staticmethod
    def forward(ctx, input1):
        output = torch.zeros(input1.size())

        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.FlowFillholelayer_gpu_forward(input1, output)
        else:
            err = my_lib.FlowFillholelayer_cpu_forward(input1, output)
        if err != 0:
            print(err)

        ctx.save_for_backward(input1.contiguous())
        ctx.device = torch.cuda.current_device() if input1.is_cuda else -1
        
        return output