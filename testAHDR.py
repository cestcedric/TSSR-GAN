import argparse
import datasets
from   submodules.AHDRNet import model
import os
import torch
from   torch.autograd import Variable
import torch.utils.data as data
from   utils import balancedsampler
from   utils import tools

class testdata(data.Dataset):
    def __init__(self, root, path_list, loader=datasets.listdatasets.rescale_loader):
        self.root = root
        self.path_list = path_list
        self.loader = loader

    def __getitem__(self, index):
        path = self.path_list[index]
        image_0, image_2, images = self.loader(root = self.root, im_path = path, max_frame_size = (1500, 1000), data_aug = False, seqlength = 3, upscaling_scale = 1)
        return image_0, image_2, images

    def __len__(self):
        return len(self.path_list)

parser = argparse.ArgumentParser(description='Attention-guided HDR')

parser.add_argument('--test_whole_Image', default='./test.txt')
parser.add_argument('--trained_model_dir', default='AHDRNet/trained-model/')
parser.add_argument('--trained_model_filename', default='ahdr_model.pt')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--load_model', default=True)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--seed', default=1)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')

args = parser.parse_args()

path_in = os.path.join('AHDRNet', 'data', 'Test')
path_out = os.path.join(path_in, 'output')

paths = [
    ';'.join([ os.path.join(path_in, 'EXTRA', '001', i) for i in ('262A2615.tif', '262A2616.tif', '262A2617.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '002', i) for i in ('262A2643.tif', '262A2644.tif', '262A2645.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '003', i) for i in ('262A2698.tif', '262A2699.tif', '262A2700.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '004', i) for i in ('262A2782.tif', '262A2783.tif', '262A2784.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '005', i) for i in ('262A2873.tif', '262A2874.tif', '262A2875.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '006', i) for i in ('262A2928.tif', '262A2929.tif', '262A2930.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '007', i) for i in ('262A3170.tif', '262A3171.tif', '262A3172.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '008', i) for i in ('262A3225.tif', '262A3226.tif', '262A3227.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '009', i) for i in ('262A3239.tif', '262A3240.tif', '262A3241.tif') ]),
    ';'.join([ os.path.join(path_in, 'EXTRA', '010', i) for i in ('262A3282.tif', '262A3283.tif', '262A3284.tif') ]),
    ';'.join([ os.path.join(path_in, 'PAPER', 'BarbequeDay', i) for i in ('262A2943.tif', '262A2944.tif', '262A2945.tif') ]),
    ';'.join([ os.path.join(path_in, 'PAPER', 'LadySitting', i) for i in ('262A2705.tif', '262A2706.tif', '262A2707.tif') ]),
    ';'.join([ os.path.join(path_in, 'PAPER', 'ManStanding', i) for i in ('262A2629.tif', '262A2630.tif', '262A2631.tif') ]),
    ';'.join([ os.path.join(path_in, 'PAPER', 'PeopleStanding', i) for i in ('262A2866.tif', '262A2867.tif', '262A2868.tif') ]),
    ';'.join([ os.path.join(path_in, 'PAPER', 'PeopleTalking', i) for i in ('262A2810.tif', '262A2811.tif', '262A2812.tif') ])
]

generator = model.AHDR(args)
tools.initialize_weights_xavier(generator)

if args.use_cuda:
    generator.cuda()
generator.eval()
testset = testdata(root = None, path_list = paths)
dataloader = torch.utils.data.DataLoader(
    testset, 
    batch_size = 1,
    sampler = balancedsampler.RandomBalancedSampler(testset, 1),
    num_workers= 2, 
    pin_memory= True
)

for i, (hr, lr) in enumerate(dataloader):
    with torch.no_grad():
        x = lr[0].cuda() if args.use_cuda else lr[0]
        y = lr[1].cuda() if args.use_cuda else lr[1]
        z = lr[2].cuda() if args.use_cuda else lr[2]
        v_x = Variable(x, requires_grad = False)
        v_y = Variable(y, requires_grad = False)
        v_z = Variable(z, requires_grad = False)

        output = generator(v_x, v_y, v_z)

        p_x = v_x.cpu().numpy()
        p_o = output.cpu().numpy()

        tools.print_tensor(path = os.path.join(path_out, str(i).zfill(2) + '_x.png'), img = tools.printable_tensor([p_x]))
        tools.print_tensor(path = os.path.join(path_out, str(i).zfill(2) + '_o.png'), img = tools.printable_tensor([p_o]))
