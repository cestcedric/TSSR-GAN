import cv2
import imagesize
import numpy
import os
from   PIL import Image
import random
import utils.tools as tools
import torch
import torch.utils.data as data


def rescale_loader(root, im_path, max_frame_size = (0, 0), data_aug = True, improved_estimation = 0, seqlength = 0, upscaling_scale = 4):
    paths = im_path.split(';')
    paths = paths[::-1] if data_aug and random.randint(0, 1) else paths
    w_max, h_max = max_frame_size
    w_img, h_img = imagesize.get(paths[0])

    if seqlength > 0 and seqlength < len(paths):
        i = random.randint(0, len(paths) - seqlength)
        paths = paths[i:i+seqlength]

    # Handle Portrait input
    if w_img < h_img:
        w_max, h_max = h_max, w_max

    # if w_max and h_max are much smaller than w_img and h_img, we might have too little movement in the frame
    if h_img > 3*h_max and w_img > 3*w_max:
        c_h = h_img // (3*h_max) + 1
        c_w = w_img // (3*w_max) + 1
        c = max(c_h, c_w)
        w_res = w_img // c
        h_res = h_img // c
        resize = True
    else:
        w_res = w_img
        h_res = h_img
        resize = False

    # Limit maximum input size for e.g. memory issues
    w = min(w_img, w_max) if not w_max == 0 else w_img
    h = min(h_img, h_max) if not h_max == 0 else h_img

    # Memory usage increased when using improved estimation
    # Example values: HR 512x512 -> LR 128x128 with 7 frames and full backprop just fits into 24GB
    # Non-quadratic frames than squares with improved estimation (duh), so might as well just stick to squares
    # Frames that are too big for improved_estimation are not affected by this
    if improved_estimation:
        ie_scale = (improved_estimation * 128) // (min(w,h) // upscaling_scale)
        if ie_scale > 1:
            w = h = min(w,h)

    assert h > 0 and w > 0
    
    # Select cutout, if training frame smaller than actual images
    w_offset = random.choice(range(0, w_res - w + 1, 4))
    h_offset = random.choice(range(0, h_res - h + 1, 4))

    if resize:
        hr_pre = [ numpy.asarray(
            Image.open(path) \
            .resize((w_res, h_res), Image.LANCZOS) \
            .crop((w_offset, h_offset, w_offset + w, h_offset + h))) 
            for path in paths ]
    else:
        hr_pre = [ numpy.asarray(
            Image.open(path) \
            .crop((w_offset, h_offset, w_offset + w, h_offset + h))) 
            for path in paths ]

    lr_pre = [ cv2.GaussianBlur(frame, (0,0), sigmaX = 1.5)[::upscaling_scale, ::upscaling_scale, :] for frame in hr_pre ]

    if data_aug:
        if random.randint(0,1):
            hr_pre = [ numpy.fliplr(im) for im in hr_pre ]
            lr_pre = [ numpy.fliplr(im) for im in lr_pre ]
        if random.randint(0,1):
            hr_pre = [ numpy.flipud(im) for im in hr_pre ]
            lr_pre = [ numpy.flipud(im) for im in lr_pre ]

    hr_pre = [ tools.im2tensor(im) for im in hr_pre ]
    lr_pre = [ tools.im2tensor(im) for im in lr_pre ]

    return hr_pre, lr_pre


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, loader=rescale_loader, max_frame_size = (0, 0), improved_estimation = 0, seqlength = 0, upscaling_scale = 4):
        self.root = root
        self.path_list = path_list
        self.max_frame_size = max_frame_size
        self.loader = loader
        self.improved_estimation = improved_estimation
        self.seqlength = seqlength
        self.upscaling_scale = upscaling_scale

    def __getitem__(self, index):
        path = self.path_list[index]
        hr, lr = self.loader(
            root = self.root, 
            im_path = path, 
            max_frame_size = self.max_frame_size,
            improved_estimation = self.improved_estimation,
            seqlength = self.seqlength,
            upscaling_scale = self.upscaling_scale)
        return hr, lr

    def __len__(self):
        return len(self.path_list)