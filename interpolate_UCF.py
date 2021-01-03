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

data_in = os.path.join('UCF','UCF101')
data_out = os.path.join('UCF', 'interpolated')
random.seed(64209)

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

subdir = os.listdir(data_in)
gen_dir = os.path.join(data_out, unique_id)
if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)

tot_timer = AverageMeter()
proc_timer = AverageMeter()
end = time.time()
for dir in subdir: 
    print(dir)
    in_dir = os.path.join(data_in, dir)
    if not os.path.exists(os.path.join(gen_dir, dir)):
        os.mkdir(os.path.join(gen_dir, dir))
    if not os.path.exists(os.path.join(gen_dir, dir + 'input')):
        os.mkdir(os.path.join(gen_dir, dir + 'input'))

    index = random.randint(0, len(os.listdir(in_dir)) - 10) # need a few frames after that, count starts at zero, and one .avi file in directory
    arguments_strFirst = os.path.join(in_dir, dir + str(index) + '.png')
    arguments_strSecond = os.path.join(in_dir, dir + str(index+4) + '.png')

    for i in range(5):
        shutil.copy2(src = os.path.join(in_dir, dir + str(index + i) + '.png'), dst = os.path.join(gen_dir, dir + 'input', dir + str(i) + '.png'))

    X0 =  torch.from_numpy(numpy.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
    X1 =  torch.from_numpy(numpy.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

    assert (X0.shape == X1.shape)

    if len(X0.shape) == 3:
        X0 = torch.unsqueeze(X0, dim=0)
        X1 = torch.unsqueeze(X1, dim=0)

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)

    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
    proc_end = time.time()
    with torch.no_grad():
        outputs = model(frame_start = X0, frame_end = X1, recurrent = True)

    proc_timer.update(time.time() - proc_end)
    tot_timer.update(time.time() - end)
    end  = time.time()
    print("*****************current image process time \t " + str(time.time()-proc_end) +"s ******************" )
    if use_cuda:
        outputs = outputs.data.cpu().numpy() if not isinstance(outputs, list) else [ item.data.cpu().numpy() for item in outputs ]
    else:
        outputs = outputs.data.numpy() if not isinstance(outputs, list) else [ item.data.numpy() for item in outputs ]

    outputs = [numpy.transpose(255.0 * item.clip(0,1.0)[0, :, :, :], (1, 2, 0)) for item in outputs]

    count = 0
    for item in outputs:
        arguments_strOut = os.path.join(gen_dir, dir, "{:0>4d}.png".format(count))
        count = count + 1
        imsave(arguments_strOut, numpy.round(item).astype(numpy.uint8))


         
