import datasets
from   datetime import datetime
import math
import numpy
import os
import signal
import sys
import threading
import torch
from   torch.autograd import grad, Variable
import torch.utils.data
from   utils.AverageMeter import AverageMeter, RunAVGMeter
import utils.balancedsampler as balancedsampler
from   utils.Logger import Logger
import utils.loss_function as loss_function
import utils.loss_perceptual as loss_perceptual
import utils.lr_scheduler as lr_scheduler
from   utils.my_args import args
from   utils import GracefulKiller
from   utils import tools

from TSSR.Generator import Generator
from TSSR.Discriminator import Discriminator

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #TODO: remove when align_corners = true in some methods

torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark # Speed up execution: only makes sense when limited number of input sizes

killer = GracefulKiller.GracefulKiller(args.save_path)

def train():
    torch.manual_seed(args.seed)
    with open(args.arg, 'w') as f:
        for k,v in sorted(vars(args).items()): print("{0}: {1}".format(k,v), file = f)

    logger = Logger(model_id = args.unique_id, experiment_id = args.tb_experiment)

    print('Run ID:', args.unique_id)

    '''
    Create model and load pretrained state
    '''
    generator = Generator(
        channel = args.channels,
        filter_size = args.filter_size,
        scale = args.upscale_scale,
        training = True,
        improved_estimation = args.improved_estimation,
        detach_estimation = args.detach_estimation,
        interpolated_frames = args.interpolationsteps)

    if args.gan:
        in_channels = 0
        in_channels += 784 * args.interpolationsteps * args.context 
        in_channels +=   8 * args.interpolationsteps * args.depth
        in_channels +=   4 * args.interpolationsteps * args.flow
        in_channels +=   3 * (args.interpolationsteps + 4) * args.spatial
        in_channels +=   9 * args.interpolationsteps * args.temporal #TODO: 15

        discriminator = Discriminator(
            in_channels = in_channels,
            training = True)

    # Save generator and discriminator when shutting down training
    killer.attach_generator(generator)
    if args.gan:
        killer.attach_discriminator(discriminator)

    if args.loss_perceptual:
        vgg = loss_perceptual.vgg19()
        vgg.eval()
        if args.use_cuda:
            vgg = vgg.cuda()

    if args.gradient_clipping > 0:
        tools.clip_gradients(model = generator, magnitude = args.gradient_clipping)

    if args.use_cuda:
        print("Turn the models into CUDA")
        generator = generator.cuda()
        if args.gan:
            discriminator = discriminator.cuda()

    '''
    Load pretrained weights
    '''
    if not args.pretrained_generator==None:
        tools.load_model_weights(model = generator, weights = args.pretrained_generator, use_cuda = args.use_cuda)

    if not args.pretrained_discriminator == None:
        tools.load_model_weights(model = discriminator, weights = args.pretrained_discriminator, use_cuda = args.use_cuda)

    if not args.pretrained_merger == None:
        tools.load_model_weights(model = generator.mergeNet, weights = args.pretrained_merger, use_cuda = args.use_cuda)

    if not args.pretrained_upscaler == None:
        tools.load_model_weights(model = generator.upscaleNet, weights = args.pretrained_upscaler, use_cuda = args.use_cuda)

    if args.perturb != 0:
        tools.perturb_weights(ctx = generator, p = args.perturb)
        tools.perturb_weights(ctx = discriminator, p = args.perturb)
    

    '''
    Load Data
    '''
    train_set, valid_set = datasets.UpscalingData(
        root = args.datasetPath,
        max_frame_size = (args.max_img_width, args.max_img_height),
        improved_estimation = args.improved_estimation,
        seqlength = args.interpolationsteps + 2,
        upscaling_scale = args.upscale_scale)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = args.batch_size,
        sampler = balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / args.batch_size )),
        num_workers = args.workers, 
        pin_memory = True 
        )
    val_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size = args.batch_size, 
        num_workers = args.workers, 
        pin_memory = True
        )
    print('{} samples found, {} train samples and {} test samples '.format(len(valid_set)+len(train_set), len(train_set), len(valid_set)))
    
    '''
    Optimizers
    '''
    print("train the interpolation net")
    # 1 step -> gradient from frame 1
    # 2 steps-> gradient from frame 1 + gradient from frame 2 + backpropagation through recurrent frame 1
    # ... => scale
    lr_scale = 2 / (args.interpolationsteps * (args.interpolationsteps + 1))
    optimizer_g = torch.optim.Adamax([
        {'params': generator.ctxNet.parameters(), 'lr': args.coe_ctx * args.lr_generator * lr_scale},
        {'params': generator.depthNet.parameters(), 'lr': args.coe_depth * args.lr_generator * lr_scale},
        {'params': generator.flownets.parameters(), 'lr': args.coe_flow * args.lr_generator * lr_scale},
        {'params': generator.initScaleNets_filter.parameters(), 'lr': args.coe_filter * args.lr_generator * lr_scale},
        {'params': generator.initScaleNets_filter1.parameters(), 'lr': args.coe_filter * args.lr_generator * lr_scale},
        {'params': generator.initScaleNets_filter2.parameters(), 'lr': args.coe_filter * args.lr_generator * lr_scale},
        {'params': generator.mergeNet.parameters(), 'lr': args.coe_merge * args.lr_generator * lr_scale},
        {'params': generator.rectifyNet.parameters(), 'lr': args.lr_rectify * lr_scale},
        {'params': generator.upscaleNet.parameters(), 'lr': args.lr_upscale * lr_scale}
        ],
        lr = args.lr_generator * lr_scale, betas = (0.9, 0.999), eps = 1e-8, weight_decay = args.weight_decay)

    if args.gan:
        optimizer_d = torch.optim.Adam(
            params = discriminator.parameters(),
            lr = args.lr_discriminator if args.loss_layer else args.lr_discriminator * 0.3,
            betas = (0.5, 0.999),
            eps = 1e-8, 
            weight_decay = args.weight_decay
        )
        if args.warmup_discriminator and args.warmup_boost:
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = param_group['lr']*args.warmup_boost
                print('Discriminator learning rate boosted to', param_group['lr'])



    scheduler_g = lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min', factor = args.factor, patience = args.patience, verbose = True)
    if args.gan:
        scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d, 'max', factor = 0.5, patience = 5, verbose = True) # Only used for kickstarting discriminator

    '''
    Output loads of stuff
    '''
    print("*********Start Training********")
    print("Generator LR is: "+ str(float(optimizer_g.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))
    
    print("Num. of generator parameters:", tools.count_network_parameters(generator))
    if hasattr(generator,'flownets'):
        print("Num. of flow model parameters:", tools.count_network_parameters(generator.flownets))
    if hasattr(generator,'initScaleNets_occlusion'):
        print("Num. of initScaleNets_occlusion model parameters:", tools.count_network_parameters(generator.initScaleNets_occlusion) + tools.count_network_parameters(generator.initScaleNets_occlusion1) + tools.count_network_parameters(generator.initScaleNets_occlusion2))
    if hasattr(generator,'initScaleNets_filter'):
        print("Num. of initScaleNets_filter model parameters:", tools.count_network_parameters(generator.initScaleNets_filter) + tools.count_network_parameters(generator.initScaleNets_filter1) + tools.count_network_parameters(generator.initScaleNets_filter2))
    if hasattr(generator, 'ctxNet'):
        print("Num. of ctxNet model parameters:", tools.count_network_parameters(generator.ctxNet))
    if hasattr(generator, 'depthNet'):
        print("Num. of depthNet model parameters:", tools.count_network_parameters(generator.depthNet))
    if hasattr(generator,'rectifyNet'):
        print("Num. of rectifyNet model parameters:", tools.count_network_parameters(generator.rectifyNet))
    if hasattr(generator, 'mergeNet'):
        print('Num. of merge network parameters:', tools.count_network_parameters(generator.mergeNet))
    if hasattr(generator,'upscaleNet'):
        print("Num. of upscaleNet model parameters:", tools.count_network_parameters(generator.upscaleNet))
    if args.gan:
        print("Num. of discriminator model parameters:", tools.count_network_parameters(discriminator))

    '''
    Define parameters and variables for stats
    '''
    t_loss_charbonnier_sr = AverageMeter()
    t_loss_charbonnier_lr = AverageMeter()
    t_loss_perceptual = AverageMeter()
    t_loss_pingpong = AverageMeter()
    t_accuracy = RunAVGMeter(val = 0, size = args.warmup_lag)
    v_loss = AverageMeter()
    v_loss_charbonnier = AverageMeter()
    v_loss_perceptual = AverageMeter()
    v_loss_best = 10e10
    dis_trained = 0
    gen_trained = 0

    # Epoch length training
    if args.lengthEpoch > len(train_set) or args.lengthEpoch < 1:
        epoch_length = int(len(train_set) / args.batch_size)
    else:
         epoch_length = int(args.lengthEpoch / args.batch_size)

    # Epoch length validation
    coe_val = (1-args.test_valid_split)
    if args.lengthEpoch * coe_val > len(valid_set) or args.lengthEpoch < 1:
        valid_length = int(len(valid_set) / args.batch_size)
    else:
         valid_length = int(args.lengthEpoch * coe_val / args.batch_size)

    # Reduce discriminator warmup time by not computing useless generator losses
    if args.warmup_discriminator:
        tmp_loss_perceptual = args.loss_perceptual
        tmp_loss_pingpong = args.loss_pingpong
        tmp_loss_layer = args.loss_layer
        args.loss_perceptual = False
        args.loss_pingpong = False
        args.loss_layer = False

    debug_root = os.path.join(os.getcwd(), args.debug_output_dir)

    def __step_train(i, hr, lr, epoch_length, epoch_dis, epoch_gen, dis_trained, gen_trained, sr_fadein, log_this_it, save_this_it):
        '''
        1 Input Handling:
            - high resolution frames to cuda
            - first and last frame of low resolution sequence to cuda
            - pad inputs with H or W < 128
        '''
        assert(len(hr) == len(lr) == (args.interpolationsteps + 2))
        hr = [ Variable(frame.cuda() if args.use_cuda else frame, requires_grad = args.log_gradients) for frame in hr]
        lr = [ Variable(frame.cuda() if args.use_cuda else frame, requires_grad = args.log_gradients) for frame in lr]

        lr_start = lr[0]
        lr_end = lr[-1]

        '''
        2.1 Recurrent interpolation:
            - frames
            - charbonnier loss
            - perceptual loss
            - ping-pong loss
        '''
        outputs_sr, outputs_lr = generator(
            frame_start = lr_start, 
            frame_end = lr_end,
            low_memory = args.low_memory)

        loss_charbonnier_sr = loss_function.charbonnier_loss(output = outputs_sr, target = hr, epsilon = args.epsilon)
        loss_charbonnier_lr = loss_function.charbonnier_loss(output = outputs_lr[1:-1], target = lr[1:-1], epsilon = args.epsilon)

        if args.loss_perceptual:
            loss_perceptual = 0
            for h, s in zip(hr, outputs_sr):
                vgg_real = vgg(x = h, normalize = True)
                vgg_fake = vgg(x = s, normalize = True)
                loss_perceptual += loss_function.cosine_similarity(vgg_real, vgg_fake)

        if args.loss_pingpong:
            #TODO: LR ping pong as well?
            outputs_sr_rev = generator(frame_start = lr_end, frame_end = lr_start, low_memory = args.low_memory)[0]
            outputs_sr_rev.reverse()
            loss_pingpong = loss_function.pingpong_loss(outputs_sr, outputs_sr_rev)
        '''
        2.2 Discriminator:
            - GAN loss
            - Layer loss
        '''
        if args.gan:
            discriminator_input_real = generator.prepare_discriminator_inputs(
                sr = hr,
                frame0 = lr[0], 
                frame1 = lr[-1], 
                temporal = args.temporal, 
                spatial = args.spatial, 
                context = args.context, 
                depth = args.depth, 
                flow = args.flow)
            discriminator_input_fake = generator.prepare_discriminator_inputs(
                sr = outputs_sr,
                frame0 = lr[0], 
                frame1 = lr[-1], 
                temporal = args.temporal, 
                spatial = args.spatial, 
                context = args.context, 
                depth = args.depth, 
                flow = args.flow)

            score_real, layers_real = discriminator(discriminator_input_real.detach())
            score_fake, _ = discriminator(discriminator_input_fake.detach())
            score_gen , layers_gen  = discriminator(discriminator_input_fake)

            loss_gen, loss_dis, loss_real, loss_fake = loss_function.gan_loss(
                score_real = score_real, 
                score_fake = score_fake, 
                score_gen  = score_gen, 
                epsilon = args.gan_epsilon,
                batchsize = args.batch_size)

            loss_layer = loss_function.layerloss(layers_real, layers_gen) if args.loss_layer else 0


        '''
        3 Loss handling:
            - weighting
            - balance generator and discriminator power
            - backpropagation
        '''
        # Combine losses
        loss_total = loss_charbonnier_lr * args.scale_lr / (args.scale_lr + args.scale_sr * sr_fadein)
        loss_total += loss_charbonnier_sr * args.scale_sr * sr_fadein / (args.scale_lr + args.scale_sr * sr_fadein)
        if args.loss_layer:
            loss_total += loss_layer * args.scale_layer * sr_fadein
        if args.loss_perceptual:
            loss_total += loss_perceptual * args.scale_perceptual * sr_fadein
        if args.loss_pingpong:
            loss_total += loss_pingpong * args.scale_pingpong * sr_fadein
        if args.gan:
            loss_total += loss_gen * args.scale_gan * sr_fadein

        # Scheduling
        if args.gan:
            gen_behindness = math.fabs(loss_gen.item()) - math.fabs(loss_real.item())
            train_dis = False if args.freeze_dis else (True if args.freeze_gen else (gen_behindness < args.dis_threshold))
            train_gen = False if args.freeze_gen else (True if args.freeze_dis else (gen_behindness > args.gen_threshold))
            dis_trained += train_dis
            gen_trained += train_gen
            epoch_dis += train_dis
            epoch_gen += train_gen

            # Backpropagation
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if train_dis:
                loss_dis.backward(retain_graph = train_gen) # intermediate layers of discriminator used for layerloss for generator
            if train_gen:
                loss_total.backward()
        else:
            optimizer_g.zero_grad()
            loss_total.backward()
            train_gen = True
                    

        '''
        4 Logging:
            - AverageMeters to print to screen
            - Losses to tensorboard
            - Gradients and weights to tensorboard
            - Save debugging frames to disk
        '''
        if args.gan:
            t_acc_real, t_acc_fake = tools.accuracy(score_real = torch.mean(score_real).item(), score_fake = torch.mean(score_fake).item())
            t_accuracy.update(val = 0.5*t_acc_real + 0.5*t_acc_fake, weight = args.batch_size)

        t_loss_charbonnier_sr.update(val = loss_charbonnier_sr.item(), n = args.batch_size)
        t_loss_charbonnier_lr.update(val = loss_charbonnier_lr.item(), n = args.batch_size)
        t_loss_perceptual.update(val = loss_perceptual.item() if args.loss_perceptual else 0, n = args.batch_size)
        t_loss_pingpong.update(val = loss_pingpong.item() if args.loss_pingpong else 0, n = args.batch_size)

        logger.log_scalars(tag = 'Generator', 
            tag_value_dict = {
                'total': loss_total.item(),
                'charbonnier': loss_charbonnier_sr.item(),
                'charbonnier_lr': loss_charbonnier_lr.item(),
                'gan': loss_gen.item() if args.gan else 0,
                'layer': loss_layer.item() if args.loss_layer else 0, 
                'pingpong': loss_pingpong.item() if args.loss_pingpong else 0, 
                'perceptual': loss_perceptual.item() if args.loss_perceptual else 0}, 
            epoch = t, n_batch = i, num_batches = epoch_length)
        if args.gan:
            logger.log_scalars(tag = 'Discriminator/loss', 
                tag_value_dict = {
                    'real': loss_real.item(), 
                    'fake': loss_fake.item(),
                    'gen_behindness': gen_behindness},
                epoch = t, n_batch = i, num_batches = epoch_length)
            logger.log_scalars(tag = 'Discriminator/scores', 
                tag_value_dict = {
                    'real': torch.mean(score_real).item(), 
                    'fake': torch.mean(score_fake).item()},
                epoch = t, n_batch = i, num_batches = epoch_length)
            logger.log_scalars(tag = 'Discriminator/detection_performance', 
                tag_value_dict = {
                    'real': t_acc_real, 
                    'fake': t_acc_fake,
                    'avg': t_accuracy.avg},
                epoch = t, n_batch = i, num_batches = epoch_length)
            logger.log_scalars(tag = 'Scheduling', 
                tag_value_dict = {
                    'train_discriminator': 1 if train_dis else 0, 
                    'train_generator': 1 if train_gen else 0, 
                    'gen_behindness': gen_behindness}, 
                epoch = t, n_batch = i, num_batches = epoch_length)
            logger.log_scalars(tag = 'Overview_trained', 
                tag_value_dict = {
                    'generator': gen_trained, 
                    'discriminator': dis_trained}, 
                epoch = t, n_batch = i, num_batches = epoch_length)
        if args.log_gradients:
            if train_gen:
                logger.log_histogram(tag = 'weights/filter', values = tools.model_parameters(generator.initScaleNets_filter, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/filter', values = tools.model_parameters(generator.initScaleNets_filter, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/filter1', values = tools.model_parameters(generator.initScaleNets_filter1, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/filter1', values = tools.model_parameters(generator.initScaleNets_filter1, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/filter2', values = tools.model_parameters(generator.initScaleNets_filter2, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/filter2', values = tools.model_parameters(generator.initScaleNets_filter2, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/ctxNet', values = tools.model_parameters(generator.ctxNet, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/ctxNet', values = tools.model_parameters(generator.ctxNet, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/flownets', values = tools.model_parameters(generator.flownets, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/flownets', values = tools.model_parameters(generator.flownets, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/depthNet', values = tools.model_parameters(generator.depthNet, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/depthNet', values = tools.model_parameters(generator.depthNet, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/rectifyNet', values = tools.model_parameters(generator.rectifyNet, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/rectifyNet', values = tools.model_parameters(generator.rectifyNet, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/mergeNet', values = tools.model_parameters(generator.mergeNet, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/mergeNet', values = tools.model_parameters(generator.mergeNet, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'weights/upscaleNet', values = tools.model_parameters(generator.upscaleNet, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/upscaleNet', values = tools.model_parameters(generator.upscaleNet, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
            if args.gan and train_dis:
                logger.log_histogram(tag = 'weights/discriminator', values = tools.model_parameters(discriminator, 'weights'), epoch = t, n_batch = i, num_batches = epoch_length)
                logger.log_histogram(tag = 'gradients/discriminator', values = tools.model_parameters(discriminator, 'gradients'), epoch = t, n_batch = i, num_batches = epoch_length)
        # Print debugging frames
        if args.debug_output and (log_this_it or i == 0):
            if args.tb_debugframes:
                logger.save_images(tag = args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '/real', 
                    image = torch.cat([ t for t in hr ], dim = 0), 
                    epoch = t, n_batch = i, num_batches = epoch_length)
                logger.save_images(tag = args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '/fake', 
                    image = torch.cat([ t for t in outputs_sr ], dim = 0),
                    epoch = t, n_batch = i, num_batches = epoch_length)
            tools.print_tensor(
                path = os.path.join(debug_root, args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '_real.png'), 
                img = tools.printable_tensor([ t.detach().cpu().numpy() for t in hr ]))
            tools.print_tensor(
                path = os.path.join(debug_root, args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '_fake.png'), 
                img = tools.printable_tensor([ t.detach().cpu().numpy() for t in outputs_sr ]))
            if args.loss_pingpong:
                if args.tb_debugframes:
                    logger.save_images(tag = args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '/rev', 
                        image = torch.cat([ t for t in outputs_sr_rev ], dim = 0),
                        epoch = t, n_batch = i, num_batches = epoch_length)
                tools.print_tensor(
                    path = os.path.join(debug_root, args.unique_id + '_' + str(t).zfill(3) + '_' + str(i).zfill(5) + '_rev.png'), 
                    img = tools.printable_tensor([ t.detach().cpu().numpy() for t in outputs_sr_rev ]))

        '''
        5 Finish:
            - optimizer step
            - save intermediate weights
        '''
        if args.gan and train_dis:
            if args.gradient_scaling > 0:
                tools.rescale_gradients(model = discriminator, magnitude = args.gradient_scaling)
            optimizer_d.step()
        if train_gen:
            if args.gradient_scaling > 0:
                tools.rescale_gradients(model = generator, magnitude = args.gradient_scaling)
            optimizer_g.step()

        # Save intermediate weights
        if save_this_it:
            torch.save(generator.state_dict(), os.path.join(args.save_path, str(t).zfill(3) + '_' + str(i).zfill(6) + "_GEN.pth"))
            if args.gan:
                torch.save(discriminator.state_dict(), os.path.join(args.save_path, str(t).zfill(3) + '_' + str(i).zfill(6) + "_DIS.pth"))

        return epoch_dis, epoch_gen, dis_trained, gen_trained
    
    def __step_validate(i, hr, lr):
        hr = [ Variable(frame.cuda() if args.use_cuda else frame, requires_grad = False) for frame in hr]
        lr_start = Variable(lr[0].cuda() if args.use_cuda else lr[0], requires_grad = False)
        lr_end = Variable(lr[-1].cuda() if args.use_cuda else lr[-1], requires_grad = False)

        outputs_sr = generator(frame_start = lr_start, frame_end = lr_end)

        loss_charbonnier = loss_function.charbonnier_loss(output = outputs_sr, target = hr, epsilon = args.epsilon)

        if args.loss_perceptual:
            loss_perceptual = 0
            for h, s in zip(hr, outputs_sr):
                vgg_real = vgg(x = h, normalize = True)
                vgg_fake = vgg(x = s, normalize = True)
                loss_perceptual += loss_function.cosine_similarity(vgg_real, vgg_fake)

        loss_total = loss_charbonnier * args.scale_sr
        if args.loss_perceptual:
            loss_total += loss_perceptual * args.scale_perceptual

        v_loss_charbonnier.update(val = loss_charbonnier.item(), n = args.batch_size)
        v_loss_perceptual.update(val = loss_perceptual.item() if args.loss_perceptual else 0, n = args.batch_size)
        v_loss.update(val = loss_total.item(), n = args.batch_size)
            

    warmup_samples = 0
    for t in range(args.numEpoch + args.warmup_discriminator): # extra epoch for discriminator pretraining at the beginning
        '''
        Training
        '''
        generator = generator.train()
        if args.gan:
            discriminator = discriminator.train()

        epoch_dis = 0
        epoch_gen = 0

        if args.warmup_discriminator:
            args.freeze_gen = True

        for i, (hr, lr) in enumerate(train_loader):
            if i >= epoch_length:
                if not args.warmup_discriminator:
                    break
            
            # Logging anything to screen this iteration?
            log_this_it = (i % max(1, int(epoch_length/args.debug_output_freq)) == 0 and i > 0)
            save_this_it = (i % max(1, args.save_interval) == 0 and args.save_interval > 0 and i > 0)
            epoch_dis, epoch_gen, dis_trained, gen_trained = __step_train(
                i = i, 
                hr = hr, 
                lr = lr, 
                epoch_length = epoch_length, 
                epoch_dis = epoch_dis, 
                epoch_gen = epoch_gen, 
                dis_trained = dis_trained,
                gen_trained = gen_trained,
                sr_fadein = min(1, t/max(1, args.sr_fadein)),
                log_this_it = log_this_it,
                save_this_it = save_this_it)

            if log_this_it:
                print('Epoch:', t, '    [', '{:6d}'.format(i),'\\', '{:6d}'.format(epoch_length), ']', \
                    'Average charbonnier loss:', '{:7.5f}'.format(t_loss_charbonnier_sr.avg), \
                    '\tAverage perceptual loss:', '{:7.5f}'.format(t_loss_perceptual.avg), \
                    '\tdis:', '{:6d}'.format(epoch_dis), \
                    '\tgen:', '{:6d}'.format(epoch_gen), \
                    '\tdiscriminator accuracy:', '{:7.5f}'.format(t_accuracy.avg))
                t_loss_charbonnier_sr.reset()
                t_loss_perceptual.reset()
                t_loss_pingpong.reset()
                epoch_dis = 0
                epoch_gen = 0

            if args.warmup_discriminator:
                warmup_samples += 1
                if warmup_samples > args.warmup_lag:
                    if warmup_samples % (epoch_length // 5) == 0:
                        scheduler_d.step(t_accuracy.avg)
                    if t_accuracy.avg > args.warmup_threshold:
                        torch.save(discriminator.state_dict(), os.path.join(args.save_path, 'pretrain_dis_' + str(warmup_samples).zfill(5) + '.pth'))
                        args.freeze_gen = False
                        args.warmup_discriminator = False
                        args.loss_perceptual = tmp_loss_perceptual
                        args.loss_pingpong = tmp_loss_pingpong
                        args.loss_layer = tmp_loss_layer
                        logger.set_offset(warmup_samples)
                        for param_group in optimizer_d.param_groups:
                            param_group['lr'] = param_group['lr']/args.warmup_boost
                        print('Warmup finished: discriminator pretrained for', warmup_samples, 'samples.')
                        print('Start training generator.')
                        break

        '''
        Validation
        '''
        print('-'*50)
        print('Start validation:', datetime.now().isoformat())
        torch.save(generator.state_dict(), os.path.join(args.save_path, "GEN_" + str(t).zfill(3) + ".pth"))
        if args.gan:
            torch.save(discriminator.state_dict(), os.path.join(args.save_path, "DIS_" + str(t).zfill(3) + ".pth"))

        generator = generator.eval()

        with torch.no_grad():
            for i, (hr, lr) in enumerate(val_loader):
                if i >= valid_length:
                    break
                __step_validate(i = i, hr = hr, lr = lr)

        if v_loss.avg < v_loss_best:
            torch.save(generator.state_dict(), os.path.join(args.save_path, "best_GEN.pth"))

        logger.log_scalars(tag = 'Validation', 
            tag_value_dict = {
                'charbonnier': v_loss_charbonnier.avg, 
                'perceptual': v_loss_perceptual.avg,
                'total': v_loss.avg}, 
            epoch = 0, n_batch = t, num_batches = args.numEpoch)

        print('-'*50)
        print('Epoch:', t)
        print('Validation loss:', '{:7.5f}'.format(v_loss.avg))
        print('Validation charbonnier loss:', '{:7.5f}'.format(v_loss_charbonnier.avg))
        print('Validation perceptual loss:', '{:7.5f}'.format(v_loss_perceptual.avg))
        if v_loss.avg < v_loss_best:
            print('Best weights updated.')
        print('-'*50)
        v_loss_best = v_loss.avg
        scheduler_g.step(v_loss.avg)
        
        v_loss.reset()
        v_loss_charbonnier.reset()
        v_loss_perceptual.reset()
    
    print('='*100)
    print('Training finished')
    print('='*100)

if __name__ == '__main__':
    assert(sys.version_info[0] >= 3)
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
