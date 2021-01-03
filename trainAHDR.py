import datasets
from   datetime import datetime
from   AHDRNet import model
import os
import torch
from   torch.autograd import Variable
import torch.utils.data as data
from   utils.my_args import args
from   utils.AverageMeter import AverageMeter
from   utils import balancedsampler
from   utils.Logger import Logger
from   utils import tools

def train():
    def _step(model, lin, rec):
        if args.merge_architecture == 0:
            return model(x1 = lin, x2 = rec)
        elif args.merge_architecture == 1:
            return model(x1 = rec, x2 = lin)
        else:
            raise RuntimeError('Merge_architecture ' + str(args.merge) + ' not supported.')

    with open(args.arg, 'w') as f:
        for k,v in sorted(vars(args).items()): print("{0}: {1}".format(k,v), file = f)
    logger = Logger(model_id = args.unique_id, experiment_id = args.tb_experiment)
    print('Run ID:', args.unique_id)

    generator = model.AHDR()

    if not args.SAVED_MODEL == None:
        tools.load_model_weights(model = generator, weights = args.SAVED_MODEL, use_cuda = args.use_cuda)
    else:
        tools.initialize_weights_xavier(generator)
    if args.use_cuda:
        generator.cuda()
        
    generator.eval()
    train_set, valid_set = datasets.UpscalingData(root = args.datasetPath, upscaling_scale = 1)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set, 
        batch_size = args.batch_size,
        sampler=balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / args.batch_size )),
        num_workers = args.workers, 
        pin_memory = True 
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_set, 
        batch_size = args.batch_size, 
        num_workers = args.workers, 
        pin_memory = True 
    )

    optimizer = torch.optim.Adam(
        params = generator.parameters(),
        lr = args.lr,
        weight_decay=args.weight_decay
    )

    lossfunction = torch.nn.L1Loss(reduction = 'mean')
    trainloss = AverageMeter()
    validloss = AverageMeter()
    bestloss = 10e10

    
    for e in range(args.numEpoch):

        print('Training epoch', e+1, 'of', args.numEpoch)
        epoch_length = len(train_set) // args.batch_size
        generator.train()
        for i, (hr, lr) in enumerate(train_loader):
            if i > epoch_length:
                break

            lin = lr[0].cuda() if args.use_cuda else lr[0]
            real = lr[1].cuda() if args.use_cuda else lr[1]
            rec = lr[2].cuda() if args.use_cuda else lr[2]

            output = _step(model = generator, lin = lin, rec = rec)

            loss = lossfunction(output, real)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainloss.update(val = float(loss), n = args.batch_size)
            logger.log_scalars(tag = 'Training', tag_value_dict = {'loss': float(loss)}, epoch = e, n_batch = i, num_batches = epoch_length)

            if i % (epoch_length//10) == 0:
                print('[', datetime.now().isoformat(), ']', 'Training loss:', trainloss.avg)
                trainloss.reset
                tools.print_tensor(path = os.path.join(args.debug_output_dir, str(e).zfill(3) + str(i).zfill(5) + '_out.png'), img = tools.printable_tensor([output.detach().cpu().numpy()]))
                tools.print_tensor(path = os.path.join(args.debug_output_dir, str(e).zfill(3) + str(i).zfill(5) + '_real.png'), img = tools.printable_tensor([real.detach().cpu().numpy()]))
                torch.save(generator.state_dict(), os.path.join(args.save_path, 'merger_' + str(e).zfill(3) + '_' + str(i).zfill(5) + '.pth'))

        epoch_length = len(train_set) // args.batch_size
        generator.eval()
        for i, (lin, rec, real) in enumerate(valid_loader):
            if i > epoch_length:
                break

            with torch.no_grad():
                lin = lin.cuda() if args.use_cuda else lin
                real = real.cuda() if args.use_cuda else real
                rec = rec.cuda() if args.use_cuda else rec

                real = torch.squeeze(real, dim = 0)

                output = _step(model = generator, lin = lin, rec = rec)

            loss = lossfunction(output, real)

            validloss.update(val = float(loss), n = args.batch_size)
            logger.log_scalars(tag = 'Validation', tag_value_dict = {'loss': float(loss)}, epoch = e, n_batch = i, num_batches = epoch_length)

        print('[', datetime.now().isoformat(), ']', 'Validation loss:', validloss.avg)
        if validloss.avg < bestloss:
            bestloss = validloss.avg
            print('Weights updated')
            torch.save(generator.state_dict(), os.path.join(args.save_path, "best_merger.pth"))
        validloss.reset


if __name__ == '__main__':
    train()

    exit(0)