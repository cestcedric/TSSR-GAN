import argparse
import glob
from   imageio import imread, imsave
import numpy
import os
import random
import shutil
import time
import torch
from   torch.autograd import Variable
import utils.tools as tools

from   TSSR.Generator import Generator

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # remove new default for align_corners warning

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default = None, help = 'Interpolating based on this dir and subdirs, one sequence per dir.')
parser.add_argument('--output_dir', type = str, default = None)
parser.add_argument('--weights', type = str, default = None)
parser.add_argument('--timestep', type = float, default = 0.5)
parser.add_argument('--improved_estimation', type = int, default = 2)
parser.add_argument('--no_benchmark', dest = 'no_benchmark', action = 'store_true')
parser.set_defaults(no_benchmark = False)
parser.add_argument('--no_gpu', dest = 'no_gpu', action = 'store_true')
parser.set_defaults(no_gpu = False)
args = parser.parse_args()

assert args.input_dir != None and args.output_dir != None
torch.backends.cudnn.benchmark = not args.no_benchmark
interpolated_frames = int(1.0/args.timestep) - 1
IMAGE_EXTENSIONS = ['jpg', 'JPG', 'png', 'PNG']

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

model = Generator(
    improved_estimation = args.improved_estimation,
    interpolated_frames = interpolated_frames)

if not args.no_gpu:
    model = model.cuda()

if args.weights == None:
    args.weights = './model_weights/TSSR_best.pth'
if os.path.exists(args.weights):
    tools.load_model_weights(model = model, weights = args.weights, use_cuda = not args.no_gpu)
else:
    print("*****************************************************************")
    print("*************** We don't load any trained weights ***************")
    print("*****************************************************************")

model = model.eval()

subdirs = sorted(os.listdir(args.input_dir))
total_duration = 0

for subdir in subdirs:
    print(subdir + ': ', end = '')
    path_in = os.path.join(args.input_dir, subdir)
    path_out = os.path.join(args.output_dir, subdir)
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    files = tools.listIMGinDIR(path_in, depth=1)
    frame_0 = tools.read_image(files[0]) if args.no_gpu else tools.read_image(files[0]).cuda()
    frame_1 = tools.read_image(files[-1]) if args.no_gpu else tools.read_image(files[-1]).cuda()

    if len(frame_0.shape) == 3:
        frame_0 = torch.unsqueeze(frame_0, dim = 0)
        frame_1 = torch.unsqueeze(frame_1, dim = 0)

    time_start = time.time()

    with torch.no_grad():
        output = model.forward(frame_start = frame_0, frame_end = frame_1)
        
        for i, frame in enumerate(output):
            f = frame.data.cpu().numpy()
            f = tools.tensor2im(f)
            tools.print_tensor(os.path.join(path_out, str(i).zfill(5) + '.png'), f)

    duration = time.time() - time_start
    print('{:2.2f}'.format(duration) + 's,', len(output), 'frames')

print('Interpolation finished.')
print('Total duration: {:2.2f}'.format(total_duration) + 's')
print('Average duration per frame: {:2.2f}'.format(total_duration/((interpolated_frames+2)*len(subdirs))) + 's')
