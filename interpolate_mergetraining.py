import datasets
from   datetime import datetime
import numpy
import os
import random
import shutil
import subprocess
import time
import torch
from   torch.autograd import Variable
import utils.balancedsampler as balancedsampler
from   utils.my_args import args
import utils.tools as tools

from   TSSR.Generator import Generator

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def createdataset(interpolated_frames, input_data, output_path, output_set_size, use_cuda, save_which, batch_size, saved_model):
    step = output_set_size // 100

    generator = Generator(interpolated_frames = interpolated_frames) 
    if use_cuda:
        generator = generator.cuda()
    tools.load_model_weights(model = generator, weights = saved_model, use_cuda = use_cuda)
    generator = generator.eval()

    dataloader = torch.utils.data.DataLoader(
        input_data, 
        batch_size = 1,
        sampler = balancedsampler.RandomBalancedSampler(input_data, int(len(input_data) / batch_size )),
        num_workers = args.workers, 
        pin_memory = not args.cuda_unpin_memory 
    )

    for i, (x0, x2, y) in enumerate(dataloader):
        if i > output_set_size:
            break

        x0 = x0.cuda() if args.use_cuda else x0
        x2 = x2.cuda() if args.use_cuda else x2
        x0_v = Variable(x0)
        x2_v = Variable(x2)

        if use_cuda:
            x0_v.cuda()
            x2_v.cuda()

        generator_input = torch.stack((x0_v, x2_v), dim = 0)

        y_lin = generator(generator_input, recurrent = False)[0][save_which]
        y_rec = generator(generator_input, recurrent = True)[0][save_which]

        y = [ numpy.expand_dims(frame.cpu().numpy(), axis = 0) for frame in y[0] ]
        y_lin = [ frame.cpu().numpy() if use_cuda else frame.numpy() for frame in y_lin ]
        y_rec = [ frame.cpu().numpy() if use_cuda else frame.numpy() for frame in y_rec ]

        for k, (real, lin, rec) in enumerate(zip(y, y_lin, y_rec)):
            tools.print_tensor(path = os.path.join(output_path, str(i).zfill(5) + str(k).zfill(2) + '_real.png'), img = tools.printable_tensor([real]))
            tools.print_tensor(path = os.path.join(output_path, str(i).zfill(5) + str(k).zfill(2) + '_lin.png'), img = tools.printable_tensor([lin]))
            tools.print_tensor(path = os.path.join(output_path, str(i).zfill(5) + str(k).zfill(2) + '_rec.png'), img = tools.printable_tensor([rec]))

        if i % step == 0:
            print('[', datetime.now().isoformat(), ']', i)


         
if __name__ == '__main__':
    print('[', datetime.now().isoformat(), '] Start dataset creation.')

    data5 = datasets.NeedForSpeed_nfs05(os.path.join(args.datasetPath, 'nfs05'))[0]
    data7 = datasets.NeedForSpeed_nfs07(os.path.join(args.datasetPath, 'nfs07'))[0]
    data9 = datasets.NeedForSpeed_nfs09(os.path.join(args.datasetPath, 'nfs09'))[0]
    data11 = datasets.NeedForSpeed_nfs11(os.path.join(args.datasetPath, 'nfs11'))[0]
    data13 = datasets.NeedForSpeed_nfs13(os.path.join(args.datasetPath, 'nfs13'))[0]

    output = args.debug_output_dir
    if not os.path.exists(output):
        os.mkdir(output)

    for i in [3, 5, 7, 9, 11]:
        p = os.path.join(output, str(i).zfill(2))
        if not os.path.exists(p):
            os.mkdir(p)

    output_set_size = 5000
    use_cuda = args.use_cuda
    save_which = 1
    batch_size = args.batch_size
    saved_model = './model_weights/pretrained/best.pth' if args.SAVED_MODEL == None else args.SAVED_MODEL


    # print('[', datetime.now().isoformat(), '] Start dataset 3 creation.')
    # createdataset(
    #     interpolated_frames = 3, 
    #     input_data = data5, 
    #     output_path = os.path.join(output, '03'), 
    #     output_set_size = output_set_size, use_cuda = use_cuda, save_which = save_which, batch_size = batch_size, saved_model = saved_model)

    # print('[', datetime.now().isoformat(), '] Start dataset 5 creation.')
    # createdataset(
    #     interpolated_frames = 5, 
    #     input_data = data7, 
    #     output_path = os.path.join(output, '05'), 
    #     output_set_size = output_set_size, use_cuda = use_cuda, save_which = save_which, batch_size = batch_size, saved_model = saved_model)

    # print('[', datetime.now().isoformat(), '] Start dataset 7 creation.')
    # createdataset(
    #     interpolated_frames = 7, 
    #     input_data = data9, 
    #     output_path = os.path.join(output, '07'), 
    #     output_set_size = output_set_size, use_cuda = use_cuda, save_which = save_which, batch_size = batch_size, saved_model = saved_model)

    print('[', datetime.now().isoformat(), '] Start dataset 9 creation.')
    createdataset(
        interpolated_frames = 9, 
        input_data = data11, 
        output_path = os.path.join(output, '09'), 
        output_set_size = output_set_size, use_cuda = use_cuda, save_which = save_which, batch_size = batch_size, saved_model = saved_model)

    print('[', datetime.now().isoformat(), '] Start dataset 11 creation.')
    createdataset(
        interpolated_frames = 11, 
        input_data = data13, 
        output_path = os.path.join(output, '11'), 
        output_set_size = output_set_size, use_cuda = use_cuda, save_which = save_which, batch_size = batch_size, saved_model = saved_model)
        
    print('[', datetime.now().isoformat(), '] Finished dataset creation.')
    exit(0)