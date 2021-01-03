# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import interpolation_cuda as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class InterpolationLayer(Function):
    def __init__(self):
        super(InterpolationLayer,self).__init__()

    @staticmethod
    def forward(ctx, input1, input2):

        assert(input1.is_contiguous())
        assert(input2.is_contiguous())


        if input1.is_cuda :
            # output = output.cuda()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.InterpolationLayer_gpu_forward(input1, input2, output)
        else:
            output = torch.cuda.FloatTensor(input1.data.size())
            my_lib.InterpolationLayer_cpu_forward(input1, input2, output)
        ctx.save_for_backward(input1, input2)

        return output

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2 = ctx.saved_tensors

        if input1.is_cuda:
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()

            err = my_lib.InterpolationLayer_gpu_backward(input1,input2,gradoutput,gradinput1,gradinput2)
            if err != 0 :
                print(err)
        else:
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(input2.size()).zero_()
            err = my_lib.InterpolationLayer_cpu_backward(input1, input2, gradoutput, gradinput1, gradinput2)
        if err != 0 :
            print(err)

        return gradinput1, gradinput2