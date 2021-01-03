# this is for wrapping the customized layer
import torch
from   torch.autograd import Function
import _ext.my_lib as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class SeparableConvLayer(Function):
    def __init__(self):
        super(SeparableConvLayer,self).__init__()

    @staticmethod
    def forward(ctx, input1, input2, input3, filtersize):
        intBatches = input1.size(0)
        intInputDepth = input1.size(1)
        intInputHeight = input1.size(2)
        intInputWidth = input1.size(3)
        intFilterSize = min(input2.size(1), input3.size(1))
        intOutputHeight = min(input2.size(2), input3.size(2))
        intOutputWidth = min(input2.size(3), input3.size(3))

        assert(intInputHeight - filtersize == intOutputHeight - 1)
        assert(intInputWidth - filtersize == intOutputWidth - 1)
        assert(intFilterSize == filtersize)

        assert(input1.is_contiguous() == True)
        assert(input2.is_contiguous() == True)
        assert(input3.is_contiguous() == True)

        output = input1.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.SeparableConvLayer_gpu_forward(input1, input2, input3, output)
        else:
            err = my_lib.SeparableConvLayer_cpu_forward(input1, input2, input3, output)
        if err != 0:
            print(err)
        
        ctx.save_for_backward(input1.contiguous(), input2.contiguous(), input3.contiguous())
        ctx.device = torch.cuda.current_device() if input1.is_cuda else -1
        ctx.filtersize = filtersize

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2, input3 = ctx.saved_tensors

        gradinput1 = torch.zeros(input1.size())
        gradinput2 = torch.zeros(input2.size())
        gradinput3 = torch.zeros(input3.size())
        if input1.is_cuda:
            gradinput1 = gradinput1.cuda(ctx.device)
            gradinput2 = gradinput2.cuda(ctx.device)
            gradinput3 = gradinput3.cuda(ctx.device)
            err = my_lib.SeparableConvLayer_gpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        else:
            err = my_lib.SeparableConvLayer_cpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)

        return gradinput1, gradinput2, gradinput3, None