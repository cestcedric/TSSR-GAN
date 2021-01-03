import separableconvflow_cuda as my_libimport torch
from   torch.autograd import Function
import warnings

class SeparableConvFlowLayer(Function):
    def __init__(self):
        warnings.warn("\nSeparable Conv Flow Layer is not precise enough for optical flow due to a divison operation")
        super(SeparableConvFlowLayer,self).__init__()

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

        flow_ouput = torch.zeros(intBatches, 2, intOutputHeight, intOutputWidth) # as a byproduct of SepConv, but no

        if input1.is_cuda:
            flow_ouput = flow_ouput.cuda()
            err = my_lib.SeparableConvFlowLayer_gpu_forward(input1, input2, input3, flow_ouput)
        else:
            err = my_lib.SeparableConvFlowLayer_cpu_forward(input1, input2, input3, flow_ouput)
        if err != 0:
            print(err)

        ctx.save_for_backward(input1.contiguous(), input2.contiguous(), input3.contiguous())
        ctx.filtersize = filtersize
        ctx.device = torch.cuda.current_device() if input1.is_cuda else -1

        return flow_ouput

    @staticmethod
    def backward(ctx, gradoutput):
        input1, input2, input3 = ctx.saved_tensors

        gradinput1 = torch.zeros(input1.size()) # the input1 has zero gradient because flow backprop. nothing to gradinput1
        gradinput2 = torch.zeros(input2.size())
        gradinput3 = torch.zeros(input3.size())
        
        if self.input1.is_cuda:
            gradinput1 = gradinput1.cuda(ctx.device)
            gradinput2 = gradinput2.cuda(ctx.device)
            gradinput3 = gradinput3.cuda(ctx.device)
            err = my_lib.SeparableConvFlowLayer_gpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)
        else:
            err = my_lib.SeparableConvFlowLayer_cpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)

        return gradinput1, gradinput2, gradinput3, None