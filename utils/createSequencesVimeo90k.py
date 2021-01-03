import argparse
from   datetime import datetime
import os
import random

def createSequences():
    random.seed(64209)

    parser = argparse.ArgumentParser(description='Create training sequences')
    parser.add_argument('--input_dir', type=str, default=None, help='Path to directory with train and test list to refactor')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to target directory for new train and test list')
    parser.add_argument('--seq_length', type=int, default=None, help='Length of result sequence length, must be =< input sequence length')
    parser.add_argument('--separator', type=str, default=';', help='Separator between sequence entries in each line (default: \';\')')
    args = parser.parse_args()

    if args.input_dir == None:
        print('No input directory specified.')
        exit(0)
    
    trainlist = open(os.path.join(args.input_dir, 'trainlist.txt')).read().splitlines()
    validlist = open(os.path.join(args.input_dir, 'testlist.txt')).read().splitlines()
    print('Training set size:', len(trainlist))
    print('Validation set size:', len(validlist))

    seq_length_org = len(trainlist[0].split(args.separator))
    if args.seq_length == None:
        args.seq_length = seq_length_org

    if seq_length_org < args.seq_length:
        raise RuntimeError('Result sequence length (' + str(args.seq_length) + ') longer than input sequence length (' + str(seq_length_org) + ')')

    trainset = []
    testset = []

    for line in trainlist:
        entries = line.split(args.separator)
        for offset in range(seq_length_org - args.seq_length + 1):
            trainset.append(args.separator.join(entries[offset:offset+args.seq_length]))

    for line in validlist:
        entries = line.split(args.separator)
        for offset in range(seq_length_org - args.seq_length + 1):
            testset.append(args.separator.join(entries[offset:offset+args.seq_length]))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, 'trainlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(trainset))
        myfile.write('\n')
    with open(os.path.join(args.output_dir, 'validlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(testset))
        myfile.write('\n')

if __name__ == '__main__':
    createSequences()
    exit(0)
