from   imageio import imread, imsave
import numpy
import os
import random
import shutil
import subprocess
import time
import torch
from   torch.autograd import Variable
from   utils.AverageMeter import AverageMeter
from   utils.my_args import args
import utils.tools as tools

from   TSSR.Generator import Generator

torch.backends.cudnn.benchmark = True

DO_MiddleBurryOther = True
MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-gen/"

MB_Other_DATA_scaled = "./MiddleBurySet/scaled/"
MB_Other_RESULT_scaled = "./MiddleBurySet/scaled-res/"

data_in = MB_Other_DATA_scaled
data_out = MB_Other_RESULT_scaled

if not os.path.exists(data_out):
    os.mkdir(data_out)

interpolated_frames = int(1.0/args.timestep) - 1

model = Generator(
    improved_estimation = args.improved_estimation,
    interpolated_frames = interpolated_frames)

if args.use_cuda:
    model = model.cuda()

if args.pretrained_generator == None:
    args.pretrained_generator = './model_weights/best.pth' # best DAIN weights untouched
if os.path.exists(args.pretrained_generator):
    print("The testing model weight is: " + args.pretrained_generator)
    tools.load_model_weights(model = model, weights = args.pretrained_generator, use_cuda = args.use_cuda)
else:
    print("*****************************************************************")
    print("*************** We don't load any trained weights ***************")
    print("*****************************************************************")

model = model.eval() # deploy mode

use_cuda = args.use_cuda
dtype = args.dtype
unique_id = args.unique_id
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()
if DO_MiddleBurryOther:
    subdir = os.listdir(data_in)
    gen_dir = os.path.join(data_out, unique_id)
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()
    for dir in subdir: 
        print(dir)
        # Filter used here to exclude directories with only frame10 and frame11 for longterm interpolation
        if False: #not dir in ['Hydrangea','Venus','Dimetrodon']:
            pass
        else:
            if not os.path.exists(os.path.join(gen_dir, dir)):
                os.mkdir(os.path.join(gen_dir, dir))
            arguments_strFirst = os.path.join(data_in, dir, "frame10.png")
            arguments_strSecond = os.path.join(data_in, dir, "frame11.png")

            X0 =  torch.from_numpy( numpy.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
            X1 =  torch.from_numpy( numpy.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

            y_ = torch.FloatTensor()

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if not channel == 3:
                continue

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

            torch.set_grad_enabled(False)
            X0 = Variable(torch.unsqueeze(X0,0))
            X1 = Variable(torch.unsqueeze(X1,0))
            # X0 = pader(X0)
            # X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()
            proc_end = time.time()
            outputs = model(frame_start = X0, frame_end = X1, recurrent = True)

            proc_timer.update(time.time() -proc_end)
            tot_timer.update(time.time() - end)
            end  = time.time()
            print("*****************current image process time \t " + str(time.time()-proc_end) +"s ******************" )
            if use_cuda:
                outputs = outputs.data.cpu().numpy() if not isinstance(outputs, list) else [ item.data.cpu().numpy() for item in outputs ]
            else:
                outputs = outputs.data.numpy() if not isinstance(outputs, list) else [ item.data.numpy() for item in outputs ]

            # outputs = [numpy.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop*4:(intPaddingTop+intHeight)*4, intPaddingLeft*4: (intPaddingLeft+intWidth)*4], (1, 2, 0)) for item in outputs]
            outputs = [numpy.transpose(255.0 * item.clip(0,1.0)[0, :, :, :], (1, 2, 0)) for item in outputs]

            count = 0
            for item in outputs:
                arguments_strOut = os.path.join(gen_dir, dir, "{:0>4d}.png".format(count))
                count = count + 1
                imsave(arguments_strOut, numpy.round(item).astype(numpy.uint8))


         
