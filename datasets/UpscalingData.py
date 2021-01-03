from  .listdatasets import ListDataset, rescale_loader
import math
import os
import random

def make_dataset(root, list_file):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    raw_im_list = raw_im_list[:-1]
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)

    return  raw_im_list

def UpscalingData(root, split = 1.0, single = False, task = 'interp', max_frame_size = (0, 0), improved_estimation = 0, seqlength = 0, upscaling_scale = 4):
    train_list = make_dataset(root, 'trainlist.txt')
    valid_list = make_dataset(root, 'validlist.txt')
    train_dataset = ListDataset(
        root, 
        train_list, 
        loader = rescale_loader,
        max_frame_size = max_frame_size,
        improved_estimation = improved_estimation,
        seqlength = seqlength,
        upscaling_scale = upscaling_scale)
    valid_dataset = ListDataset(
        root, 
        valid_list, 
        loader = rescale_loader,
        max_frame_size = max_frame_size,
        improved_estimation = improved_estimation,
        seqlength = seqlength,
        upscaling_scale = upscaling_scale)
    return train_dataset, valid_dataset