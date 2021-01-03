# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import interpolationch_cuda as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class InterpolationChLayer(Function):
    def __init__(self,ch):
        super(InterpolationChLayer,self).__init__()
        self.ch = ch

    @staticmethod
    def forward(ctx, input1, input2):

        assert(input1.is_contiguous())
        assert(input2.is_contiguous())

        if input1.is_cuda :
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.InterpolationChLayer_gpu_forward(input1, input2, output)
        else:
            output = torch.FloatTensor().resize_(input1.size()).zero_()
            my_lib.InterpolationChLayer_cpu_forward(input1, input2, output)

        ctx.save_for_backward(input1, input2)
        
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2 = ctx.saved_tensors

        if input1.is_cuda:
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()

            err = my_lib.InterpolationChLayer_gpu_backward(input1,input2,gradoutput,gradinput1,gradinput2)
            if err != 0 :
                print(err)
        else:
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(input2.size()).zero_()

            err = my_lib.InterpolationChLayer_cpu_backward(input1, input2, gradoutput, gradinput1, gradinput2)
            if err != 0 :
                print(err)

        return gradinput1, gradinput2