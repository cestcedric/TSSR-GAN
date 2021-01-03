import argparse
from   datetime import datetime
import os
import random

def createSequences():
    random.seed(64209)

    parser = argparse.ArgumentParser(description='Create training sequences')
    parser.add_argument('--input_path', type=str, default='input', help='output directory (default: ./input)')
    parser.add_argument('--output_path', type=str, default='output', help='output directory (default: ./output)')
    parser.add_argument('--dataset_size', type=int, default=5000, help='Last index in numbered file names (start from 0; default = 5000)')
    parser.add_argument('--interpolated_frames', type=int, default=0, help='How many frames where interpolated per step (default: 0)')
    parser.add_argument('--split', type=float, default=0.9, help='Train/Test split (default: 0.9)')
    args = parser.parse_args()

    root = args.output_path
    path_in = args.input_path

    if not os.path.exists(root):
        os.mkdir(root)

    size = args.dataset_size
    train_test_split = args.split
    trainlist = []
    validlist = []

    for i in range(size+1):
        p = os.path.join(path_in, str(i).zfill(5))
        for k in range(args.interpolated_frames):
            entry = [ p + str(k).zfill(2) + '_' + frame + '.png' for frame in ('lin', 'rec', 'real') ]
            if random.random() < train_test_split:
                trainlist.append(';'.join(entry))
            else:
                validlist.append(';'.join(entry))

    with open(os.path.join(root, 'trainlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(trainlist))
        myfile.write('\n')
    with open(os.path.join(root, 'validlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(validlist))
        myfile.write('\n')

if __name__ == '__main__':
    createSequences()
    exit(0)