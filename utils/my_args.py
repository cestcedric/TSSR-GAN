import argparse
import datetime
import numpy
import os
import shutil
import torch

parser = argparse.ArgumentParser(description='TSSR-GAN Training')

# ID of trained model
parser.add_argument('--uid', type = str, default = None, help = 'unique id for the training')
parser.add_argument('--force', action = 'store_true', help = 'force to override the given uid')
parser.set_defaults(force = False)

# Data and interpolation target
parser.add_argument('--datasetPath', default = '', help = 'the path of selected datasets')
parser.add_argument('--interpolationsteps', type = int, default = 3, help = 'Steps to be interpolated = Number of images per training sequence - 2 (default 3)')
parser.add_argument('--upscale_scale', type = int, default = 4, help ='Upscaling by upscale_scale (default 4)')
parser.add_argument('--test_valid_split', type = float, default = 0.8, help='Train/Validation split (default: 0.8/0.2)')

# Basic training settings
parser.add_argument('--seed', type = int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--numEpoch', '-e', type = int, default = 100, help = 'Number of epochs to train(default:100)')
parser.add_argument('--lengthEpoch', type = int, default = 0, help = 'Set number of frames per epoch, 0 --> all available frames (default: 0)')

# Image settings
parser.add_argument('--channels', '-c', type = int, default = 3, choices = [1,3], help ='channels of images (default:3)')
parser.add_argument('--max_img_height', type = int, default = 0, help = 'Maximum height of input frame (limit memory usage)')
parser.add_argument('--max_img_width', type = int, default = 0, help = 'Maximum width of input frame (limit memory usage)')

# PyTorch settings: 
parser.add_argument('--batch_size', '-b', type = int , default = 1, help = 'batch size (default:1)' )
parser.add_argument('--workers', '-w', type =int, default = 8, help = 'parallel workers for loading training samples (default : 1.6*10 = 16)')
parser.add_argument('--use_cuda', default = 1, type = int, help ='use cuda or not')
parser.add_argument('--use_cudnn', default = 1, type = int, help = 'use cudnn or not')
parser.add_argument('--no_cudnn_benchmark', dest = 'no_cudnn_benchmark', action = 'store_true', help = 'Deactivate cudNN benchmark mode')
parser.set_defaults(no_cudnn_benchmark = False)

# Learning rates
parser.add_argument('--lr_generator', type = float, default = 0.00002, help = 'the basic learning rate for the generator (default: 0.00002)')
parser.add_argument('--lr_rectify', type =float, default = 0.0004, help = 'the learning rate for rectify/refine subnetworks (default: 0.0004)')
parser.add_argument('--lr_upscale', type = float, default = 0.0004, help = 'upscale module learning rate (default: 0.0004)')
parser.add_argument('--lr_discriminator', type = float, default = 0.000025, help = 'discriminator learning rate (default: 0.00025)')
parser.add_argument('--coe_ctx', type = float, default = 1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--coe_depth', type = float, default = 0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--coe_filter', type = float, default = 1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--coe_flow', type = float, default = 0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--coe_merge', type = float, default = 1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')

# Additional training settings
parser.add_argument('--dis_threshold', type = float, default = 0.4, help = 'Train discriminator if gen_fake_loss - dis_real_loss < threshold (default: 0.4)')
parser.add_argument('--gen_threshold', type = float, default = 0.0, help = 'Train generator if gen_fake_loss - dis_real_loss > threshold (default: 0.0)')
parser.add_argument('--gradient_clipping', type = float, default = 0.0, help = 'Clip gradients at this value (default: 0.0 -> no gradient clipping)')
parser.add_argument('--gradient_scaling', type = float, default = 0.0, help = 'Rescale maximum gradient value to this (default: 0.0 -> no gradient scaling)')
parser.add_argument('--epsilon', type = float, default = 1e-6, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--gan_epsilon', type = float, default = 1e-12, help = 'the epsilon for GAN loss (default: 1e-12)')
parser.add_argument('--weight_decay', type = float, default = 0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type = int, default = 3, help = 'the patience of reduce on plateau')
parser.add_argument('--factor', type = float, default = 0.5, help = 'the factor of reduce on plateau')

# Generator settings
parser.add_argument('--improved_estimation', type = int, default = 2, help = 'Improve optical flow estimation for frame interpolation by using bicubic upscaling on small inputs (0: deactivated, 1: input width/heigth =< 128/ie_scale, 2: input width/heigth =< 256/ie_scale (default))')
parser.add_argument('--detach_estimation', dest = 'detach_estimation', action = 'store_true', help = 'Detach estimations for linear interpolation and upscaling (default: False)')
parser.set_defaults(detach_estimation = False)
parser.add_argument('--low_memory', dest = 'low_memory', action = 'store_true', help = 'Reduce memory consumption by limited backpropagation between frames (default: False)')
parser.set_defaults(low_memory = False)
parser.add_argument('--filter_size', '-f', type = int, default = 4, choices = [2, 4, 6, 5, 51], help = 'the size of filters used (default: 4)')
parser.add_argument('--dtype', default = torch.cuda.FloatTensor, choices = [torch.cuda.FloatTensor, torch.FloatTensor], help = 'tensor data type')

# Discriminator input
parser.add_argument('--gan', dest = 'gan', action = 'store_true', help = 'Use GAN training (default: False)')
parser.set_defaults(gan = False)
parser.add_argument('--context', dest = 'context', action = 'store_true', help = 'Use context network output in discriminator (default: False)')
parser.set_defaults(context = False)
parser.add_argument('--depth', dest = 'depth', action = 'store_true', help = 'Use depth estimation output in discriminator (default: False)')
parser.set_defaults(depth = False)
parser.add_argument('--flow', dest = 'flow', action = 'store_true', help = 'Use estimated flow in discriminator (default: False)')
parser.set_defaults(flow = False)
parser.add_argument('--temporal', dest = 'temporal', action = 'store_true', help = 'Use temporal discriminator (default: False)')
parser.set_defaults(temporal = False)
parser.add_argument('--spatial', dest = 'spatial', action = 'store_true', help = 'Use spatial discriminator (default: False)')
parser.set_defaults(spatial = False)

# Loss configuration
parser.add_argument('--scale_gan', type = float, default = 0.01, help = 'Scale GAN loss (default: 0.01)')
parser.add_argument('--loss_layer', dest = 'loss_layer', action = 'store_true', help = 'Use layerloss for generator (default: False)')
parser.set_defaults(loss_layer = False)
parser.add_argument('--scale_layer', type = float, default = 1.0, help = 'Scale layer loss (default: 1.0)')
parser.add_argument('--loss_perceptual', dest = 'loss_perceptual', action = 'store_true', help = 'Use perceptual loss for generator (default: False)')
parser.set_defaults(loss_perceptual = False)
parser.add_argument('--scale_perceptual', type = float, default = 0.075, help = 'Scale perceptual loss (default: 0.075)')
parser.add_argument('--loss_pingpong', dest = 'loss_pingpong', action = 'store_true', help = 'Use ping-pong loss for generator (default: False)')
parser.set_defaults(loss_pingpong = False)
parser.add_argument('--scale_pingpong', type = float, default = 1.0, help = 'Scale ping-pong loss (default: 1.0)')
parser.add_argument('--scale_lr', type = float, default = 1.0, help = 'Scale L1 loss computed on LR frames (default: 1.0)')
parser.add_argument('--scale_sr', type = float, default = 1.0, help = 'Scale L1 loss computed on SR frames (default: 1.0)')
parser.add_argument('--sr_fadein', type = int, default = 4, help = 'Fade in losses computed SR frames over epochs (default: 4)')

# Debugging settings
parser.add_argument('--log_gradients', dest = 'log_gradients', action = 'store_true', help = 'Create gradient histograms in tensorboard (default: False)')
parser.set_defaults(log_gradients = False)
parser.add_argument('--debug_output', dest = 'debug_output', action = 'store_true', help = 'Output of real and fake images at regular intervals (default: False)')
parser.set_defaults(debug_output = False)
parser.add_argument('--debug_output_freq', type = float, default = 10.0, help = 'Output frequency of debug images (default: 10.0 -> 10 outputs per epoch')
parser.add_argument('--debug_output_dir', type = str, default = 'training_output', help='Directory to output debug frames (default: ./training_output) !HAS TO BE CREATED BEFORE RUNNING!')
parser.add_argument('--tb_experiment', default = 'unsorted', type = str, help = 'Experiment where tensorboard data is added to (default: \'unsorted\')')
parser.add_argument('--tb_debugframes', dest = 'tb_debugframes', action = 'store_true', help = 'Save debug frames to tensorboard as well (default: False & --debug_output)')
parser.set_defaults(tb_debugframes = False)
parser.add_argument('--save_interval', type = int, default = 0, help = 'Save model weights even during training in intervals of n iterations, not only after validation (default: 0 / off)')

# Preload weights and set warmup phases or freeze generator and discriminator
parser.add_argument('--warmup_discriminator', dest = 'warmup_discriminator', action = 'store_true', help = 'Freeze generator initially to pretrain discriminator until 0.75 detection accuracy')
parser.set_defaults(warmup_discriminator = False)
parser.add_argument('--warmup_boost', type = float, default = 1.0, help = 'Boost factor for discriminator learning rate for warmup (default: 1.0)')
parser.add_argument('--warmup_threshold', type = float, default = 0.5, help = 'Threshold to stop discriminator warmup (0 means =< maximum of 0.5 prediction, 0.5 means 0.75 prediction confidence) (default: 0.5)')
parser.add_argument('--warmup_lag', type = int, default = 500, help = 'Number of samples for running average of discriminator prediction quality (default: 500)')
parser.add_argument('--pretrained_generator', dest = 'pretrained_generator', default = None, help ='Path to the pretrained model weights for generator')
parser.add_argument('--pretrained_discriminator', dest = 'pretrained_discriminator', default = None, help ='Path to the pretrained model weights for discriminator')
parser.add_argument('--pretrained_merger', type = str, default = None, help = 'Path to weights for merger')
parser.add_argument('--pretrained_upscaler', type = str, default = None, help = 'Path to weights for upscaler')
parser.add_argument('--pretrained_flowconv', type = str, default = None, help = 'Path to weights for flow conversion from TecoGAN to DAIN')
parser.add_argument('--perturb', type = float, default = 0, help = 'Perturb pretrained weights by this factor (Default: 0 = off)')
parser.add_argument('--freeze_gen', dest = 'freeze_gen', action = 'store_true', help = 'Freeze generator, train discriminator (default: False)')
parser.set_defaults(freeze_gen = False)
parser.add_argument('--freeze_dis', dest = 'freeze_dis', action = 'store_true', help = 'Freeze discriminator, train generator (default: False)')
parser.set_defaults(freeze_dis = False)

# Bonus arguments for interpolation
parser.add_argument('--timestep', type = float, default = 0.5, help = 'Timestep between frames (default: 0.5 => interpolate 1 intermediate frame)')

args = parser.parse_args()


if args.uid == None:
    timestamp = datetime.datetime.now()
    unique_id = str(timestamp.month).zfill(2) + str(timestamp.day).zfill(2) + str(timestamp.hour).zfill(2) + str(timestamp.minute).zfill(2) + str(timestamp.second).zfill(2) + '_' + str(numpy.random.randint(0, 100000)).zfill(5)
    save_path = './model_weights/' + unique_id
else:
    unique_id = args.uid
    save_path = './model_weights/'+ str(args.uid)

if not os.path.exists(save_path + "/best"+".pth"):
    os.makedirs(save_path,exist_ok=True)
else:
    if not args.force:
        raise("please use another uid ")
    else:
        print("override this uid" + args.uid)
        for m in range(1,10):
            if not os.path.exists(save_path + "/log.txt.bk" + str(m)):
                shutil.copy(save_path + "/log.txt", save_path + "/log.txt.bk" + str(m))
                shutil.copy(save_path + "/args.txt", save_path + "/args.txt.bk" + str(m))
                break


parser.add_argument('--unique_id', default = unique_id, help = 'Hack, don\'t set that variable.')
parser.add_argument('--save_path', default = save_path, help = 'the output dir of weights')
parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()

if args.use_cudnn:
    print("cudnn is used")
    torch.backends.cudnn.benchmark = True
else:
    print("cudnn is not used")
    torch.backends.cudnn.benchmark = False

# Check argument compatibility:
#   - Layerloss but no GAN is not working, as GAN layers needed for that
#   - Specifying scale for e.g. perceptual loss, but not activating perceptual loss => set scale to 0
if not args.gan:
    if not args.pretrained_discriminator == None:
        print('Provided pretrained discriminator state but specified training without discriminator: training continues without discriminator')
        print('add --gan for GAN training')
        args.pretrained_discriminator = None
    if args.context or args.depth or args.flow or args.spatial or args.temporal:
        print('Discriminator input specified, but training without discriminator selected: training continues without discriminator')
        print('add --gan for GAN training')
    if args.loss_layer:
        print('Layerloss uses discriminator activations, but training without discriminator selected: training continues without layerloss')
        print('add --gan for GAN training')
        args.loss_layer = False

if args.freeze_gen:
    if args.loss_perceptual or args.loss_pingpong or args.loss_layer:
        print('Generator frozen, but special generator losses activated.')
        print('Ping-Pong loss, perceptual loss and layer loss will be deactivated for faster execution and reduced memory consumption.')
        args.loss_perceptual = False
        args.loss_pingpong = False
        args.loss_layer = False

args.scale_layer *= args.loss_layer
args.scale_perceptual *= args.loss_perceptual
args.scale_pingpong *= args.loss_pingpong

if args.scale_lr == 0:
    args.sr_fadein = 1