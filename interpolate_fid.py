from   imageio import imread, imsave
import numpy
import os
import random
import shutil
import subprocess
import time
import torch
from   utils.my_args import args
import utils.tools as tools
from   torch.autograd import Variable

from   TSSR.Generator import Generator

torch.backends.cudnn.benchmark = True

input = os.path.join(os.sep, 'storage', 'user', 'starkc', 'masterarbeit', 'TecoDAIN', 'Vimeo90k', 'vimeo_triplet', 'sequences')
output = os.path.join(os.sep, 'storage', 'user', 'starkc', 'masterarbeit', 'Metrics', 'data', args.unique_id)

if not os.path.exists(output):
    os.mkdir(output)

interpolated_frames = int(1.0/args.time_step) - 1

model = Generator(interpolated_frames = interpolated_frames)

if args.use_cuda:
    model = model.cuda()

if args.SAVED_MODEL == None:
    args.SAVED_MODEL = './model_weights/best.pth' # best DAIN weights untouched
    tools.load_model_weights(model = model, weights = args.SAVED_MODEL, use_cuda = args.use_cuda)
if os.path.exists(args.SAVED_MODEL):
    tools.load_model_weights(model = model, weights = args.SAVED_MODEL, use_cuda = args.use_cuda)

model = model.eval() # deploy mode

use_cuda = args.use_cuda
save_which = args.save_which
dtype = args.dtype
unique_id = args.unique_id + ('_rec' if args.recurrent_interpolation else '_lin')
print("The unique id for current testing is: " + str(unique_id))

sets = int(4000/79/interpolated_frames)

for i in range(1, 79):
    dir = str(i).zfill(5)

    for j in range(1, sets):
        subdir = str(j).zfill(4)

        arguments_strFirst = os.path.join(input, dir, subdir, "im1.png")
        arguments_strSecond = os.path.join(input, dir, subdir, "im3.png")

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
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        y_s, offsets, filters = model(torch.stack((X0, X1), dim = 0), recurrent = args.recurrent_interpolation)
        y_ = y_s[save_which]
        offsets = tools.flat_list(offsets)
        filters = tools.flat_list(filters)

        if use_cuda:
            X0 = X0.data.cpu().numpy()
            X1 = X1.data.cpu().numpy()
            y_ = y_.data.cpu().numpy() if not isinstance(y_, list) else [ item.data.cpu().numpy() for item in y_ ]
            filters = [ filter_i.data.cpu().numpy() for filter_i in filters ]
            offsets = [ offset_i.data.cpu().numpy() for offset_i in offsets ]
        else:
            X0 = X0.data.numpy()
            X1 = X1.data.numpy()
            y_ = y_.data.numpy() if not isinstance(y_, list) else [ item.data.numpy() for item in y_ ]
            filters = [ filter_i.data.numpy() for filter_i in filters ]
            offsets = [ offset_i.data.numpy() for offset_i in offsets ]

        X0 = numpy.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop : intPaddingTop+intHeight, intPaddingLeft : intPaddingLeft+intWidth], (1, 2, 0))
        X1 = numpy.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop : intPaddingTop+intHeight, intPaddingLeft : intPaddingLeft+intWidth], (1, 2, 0))
        y_ = [numpy.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]

        offsets = [numpy.transpose(offset_i[0, :, intPaddingTop : intPaddingTop+intHeight, intPaddingLeft : intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offsets]
        filters = [numpy.transpose(filter_i[0, :, intPaddingTop : intPaddingTop+intHeight, intPaddingLeft : intPaddingLeft+intWidth], (1, 2, 0)) for filter_i in filters]

        timestep = args.time_step
        numFrames = int(1.0 / timestep) - 1
        time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]

            # Write intermediate, first and last frames to subdir
        count = 1
        for item, time_offset in zip(y_, time_offsets):
            arguments_strOut = os.path.join(output, dir + subdir + str(count) + '.png')
            imsave(arguments_strOut, numpy.round(item).astype(numpy.uint8))
            count += 1

    print(dir)
         
