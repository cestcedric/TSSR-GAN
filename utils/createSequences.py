import argparse
from   datetime import datetime
import os
import random

def createSequences():
    random.seed(64209)

    parser = argparse.ArgumentParser(description='Create training sequences')
    parser.add_argument('--frames', type=int, default=7, help='number of frames per sequence (default: 7)')
    parser.add_argument('--input_fps', type=int, default=30, help='FPS of input data (default: 30)')
    parser.add_argument('--output_fps', type=int, default=30, help='FPS of resampled output, must be >= input_fps (default: 30)')
    parser.add_argument('--input_path', type=str, default='input', help='output directory (default: ./input)')
    parser.add_argument('--output_path', type=str, default='output', help='output directory (default: ./output)')
    parser.add_argument('--split', type=float, default=0.9, help='Train/Test split (default: 0.9)')
    args = parser.parse_args()

    resample_rate = args.input_fps // args.output_fps

    root = args.output_path
    path_in = args.input_path

    if not os.path.exists(root):
        os.mkdir(root)

    img_per_set = args.frames
    train_test_split = args.split
    trainlist = []
    validlist = []

    dir = 1
    subfolders = [f.path for f in os.scandir(path_in) if f.is_dir()]

    for subfolder in subfolders:
        print('[', datetime.now(), ']', '\t[', 'start', ']\t', subfolder)
        dir_path_in = subfolder
        content = len(os.listdir(dir_path_in))
        if content < img_per_set:
            exit(0)

        entry_index = 1
        for i in range(1, content-img_per_set*resample_rate):
            entry = [ os.path.join(dir_path_in, str(i+j*resample_rate).zfill(5) + '.jpg') for j in range(img_per_set) ]
            if random.random() < train_test_split:
                trainlist.append(';'.join(entry))
            else:
                validlist.append(';'.join(entry))
            entry_index += 1
            if entry_index % 1000 == 1:
                print('[', datetime.now(), ']', '\t[', entry_index, ']\t', subfolder)

        print('[', datetime.now(), ']', '\t[', 'split', ']\t', subfolder)
        dir += 1


    with open(os.path.join(root, 'trainlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(trainlist))
        myfile.write('\n')
    with open(os.path.join(root, 'validlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(validlist))
        myfile.write('\n')

if __name__ == '__main__':
    createSequences()
    exit(0)