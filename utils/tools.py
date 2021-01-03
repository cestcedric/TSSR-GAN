import imageio
import math
import numpy
import os
import pathlib
import torch
from   torch._six import inf
import torch.nn as nn
import torch.nn.functional as F

'''
List all image files in directory, ignoring subdirectories
'''
IMAGE_EXTENSIONS = ['jpg', 'JPG', 'png', 'PNG']
def listIMGinDIR(dirpath, depth = 1):
    dirpath = pathlib.Path(dirpath)
    filelist = [str(file) for ext in IMAGE_EXTENSIONS for file in dirpath.glob('*.{}'.format(ext))]
    if depth == 2:
        filelist += [str(file) for ext in IMAGE_EXTENSIONS for file in dirpath.glob('*/*.{}'.format(ext))]
    elif depth > 2:
        raise NotImplementedError('Only supports search in directory in direct subdirectory.')
    return sorted(filelist)


'''
Returns flat list, no matter how deep it's nested
'''
def flat_list(inputlist):
    if not isinstance(inputlist, list):
        return [inputlist]
    outputlist = []
    for item in inputlist:
        if not isinstance(item, list):
            outputlist.append(item)
        else:
            outputlist.extend(flat_list(item))
    return outputlist


'''
Stacks tensors vertically and takes care of taking from CPU, ...
'''
def printable_tensor(input):
    frames = None
    stack = False
    for frame in input:
        i_frame = tensor2im(frame)
        if stack:
            frames = numpy.vstack((frames, i_frame))
        else:
            frames = i_frame
            stack = True
    return frames


'''
Print printable tensor to .PNG file
'''
def print_tensor(path, img):
    imageio.imwrite(uri = path, im = numpy.round(img).astype(numpy.uint8), format = 'png')


'''
Return normalized tensor
'''
def normalize_tensor(input, eps = 1e-12):
    norm_factor = torch.sqrt(torch.sum(input**2, dim = 1, keepdim = True))
    return input/(norm_factor + eps)


'''
Read image file to tensor
'''
def read_image(path):
    im = imageio.imread(path)
    im = numpy.transpose(im, (2,0,1))
    im = im.astype('float32')
    im = im/255.0
    return torch.from_numpy(im).float()


'''
Transform image format to tensor format
'''
def im2tensor(im):
    return numpy.transpose(im, (2,0,1)).astype('float32')/255.0


'''
Transform tensor format to image format
'''
def tensor2im(t):
    return numpy.transpose(255.0 * t.clip(0,1.0)[0, :, :, :], (1, 2, 0))


'''
Prints tree of list lengths, tensor shapes of tensors, list of tensors, lists of lists of tensors, ...
'''
def debug_tensor_list(tensorname, tensorlist):
    def dtl_rec(item, i):
        indent = (('. '*i) if i > 0 else '. ') + '|_ '*(i>0)
        if isinstance(item, list):
            print(indent, 'list ', len(item), sep='')
            for subitem in item:
                dtl_rec(subitem, i+1)
        elif torch.is_tensor(item) and item.dim() > 0:
            print(indent, 'tensor ', item.shape, sep='')
        elif torch.is_tensor(item) and item.dim() == 0:
            print(indent, item, sep='')
        elif isinstance(item, (numpy.ndarray, numpy.generic) ):
            print(indent, 'numpy ', item.shape, sep='')
        elif isinstance(item, float) or isinstance(item, int) or isinstance(item, str):
            print(indent, item, sep='')
        elif item == None:
            print(indent, 'None', sep='')
        else:
            print(indent, 'other object', sep='')
    print('-----------', tensorname, '-----------')
    dtl_rec(tensorlist, 0)
    print('='*50)


'''
Return number of trainable parameters
'''
def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([numpy.prod(p.size()) for p in parameters])


'''
Return all gradients and weights with parameters
'''
def model_parameters(model, name):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if name == 'weights':
        values = [ p.data for p in parameters ]
    elif name == 'gradients':
        values = [ p.grad for p in parameters ]
    else:
        raise RuntimeError('Unknown parameter')
    return values


'''
Initialize model weights, assuming that nonlinearities are ReLU
'''
def initialize_weights(ctx):
    for m in ctx.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


'''
Initialize model weights, assuming that nonlinearities are sigmoid
Original DAIN implementation, used for AHDRNet
'''
def initialize_weights_xavier(ctx):
    for m in ctx.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


'''
Perturb weights by p
'''
def perturb_weights(ctx, p):
    for m in ctx.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data = torch.FloatTensor(m.weight.size()).uniform_(1-p, 1+p) * m.weight.data
            m.bias.data = torch.FloatTensor(m.bias.size()).uniform_(1-p, 1+p) * m.bias.data


'''
Load model weights
'''
def load_model_weights(model, weights, use_cuda):
    print("Loading weights:", weights)
    if not use_cuda:
        pretrained_dict = torch.load(weights, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(weights)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    pretrained_dict = None


'''
Return list of tensors all detached
'''
def detach_tensors(tensorlist):
    return [ tensor.detach() for tensor in tensorlist ]


'''
Register gradient clipping as hook, no need to call every iteration
'''
def clip_gradients(model, magnitude):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for p in parameters:
        p.register_hook(lambda grad: torch.clamp(grad, -magnitude, magnitude))


'''
Rescale gradients between 0 and magnitude
'''
def rescale_gradients(model, magnitude):
    torch.nn.utils.clip_grad_norm_(model.parameters(), magnitude, norm_type = inf)


'''
Return a list of masks, one mask per interpolated frame
'''
def masks(t1_list, t2_list, frames):
    mask = [ t1 == t2 for t1, t2 in zip(t1_list, t2_list) ]
    if len(mask) < frames:
        return mask*frames
    return mask


'''
Apply mask to tensor and scale values, rest remains unchanged
'''
def mask_loss(t, mask, c):
    t[mask] = t[mask]*c
    return t


'''
Pad tensors if width or height not divisible by 'value'
Returns padded tensor and pad, for easy padding removal
'''
def pad_tensor(tensor, value = 128, mode = 'constant'):
    pad_w = (value - (tensor.shape[-1] % value)) % value
    pad_h = (value - (tensor.shape[-2] % value)) % value
    if pad_h == pad_w == 0:
        return tensor, None
    pad = [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2]
    return F.pad(input = tensor, pad = pad, mode = mode), pad


'''
Remove padding added using tools.pad
'''
def unpad_tensor(tensor, pad = None, scaling = 1):
    if pad == None:
        return tensor
    if scaling > 1:
        pad = [ int(p*scaling) for p in pad ]
    if pad[1] > 0 and pad[3] > 0:
        return tensor[..., pad[2]:-pad[3], pad[0]:-pad[1]]
    if pad[1] > 0:
        return tensor[..., pad[2]:, pad[0]:-pad[1]]
    if pad[3] > 0:
        return tensor[..., pad[2]:-pad[3], pad[0]:]
    return tensor


'''
Upscale a tensor
'''
def upscale(item, mode = 'bilinear', align_corners = True, scale_factor = 1):
    if isinstance(item, list):
        return [ upscale(item = entry, mode = mode, align_corners = align_corners, scale_factor = scale_factor) for entry in item ]
    if isinstance(item, tuple):
        tuple_upscaled = ()
        for entry in item:
            tuple_upscaled += (upscale(item = entry, mode = mode, align_corners = align_corners, scale_factor = scale_factor),)
        return tuple_upscaled
    if mode == 'nearest':
        return nn.Upsample(scale_factor = scale_factor, mode = mode)(item)
    return nn.Upsample(scale_factor = scale_factor, mode = mode, align_corners = align_corners)(item)


'''
Downscale a tensor
'''
def downscale(item, mode = 'bilinear', align_corners = True, scale_factor = 0):
        if isinstance(item, list):
            return [ downscale(entry, mode = mode, align_corners = align_corners, scale_factor = scale_factor) for entry in item ]
        if isinstance(item, tuple):
            tuple_downscaled = ()
            for entry in item:
                tuple_downscaled += (downscale(entry, mode = mode, align_corners = align_corners, scale_factor = scale_factor),)
            return tuple_downscaled
        return nn.functional.interpolate(item, scale_factor = scale_factor, mode = mode, align_corners = align_corners)


'''
Good: on half of potential score towards correct result
Better: closer to maximum/minimum score
Bad: on wrong half of potential scores
'''
def accuracy(score_real, score_fake):
    acc_real = max(0.0, (score_real - 0.5)*2)
    acc_fake = max(0.0, ((1.0 - score_fake) - 0.5)*2)
    return acc_real, acc_fake