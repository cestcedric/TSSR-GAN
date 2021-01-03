import argparse
import os
import random

def mix():

    parser = argparse.ArgumentParser(description='Create training sequences')
    parser.add_argument('--dataset_1', type=str, default=None, help='Path to first dataset (directory must contain trainlist.txt and validlist.txt')
    parser.add_argument('--dataset_2', type=str, default=None, help='Path to second dataset (directory must contain trainlist.txt and validlist.txt')
    parser.add_argument('--rate_1', type=int, default=1, help='Relative rate of appearance of dataset 1 in mixed dataset (default: 1)')
    parser.add_argument('--rate_2', type=int, default=1, help='Relative rate of appearance of dataset 2 in mixed dataset (default: 1)')
    parser.add_argument('--output_dir', type=str, default=None, help='Path for output dataset')
    args = parser.parse_args()

    if args.dataset_1 == None:
        print('Dataset 1 not specified.')
        exit(0)
    if args.dataset_2 == None:
        print('Dataset 2 not specified.')
        exit(0)
    if args.output_dir == None:
        print('Output dataset path not specified.')
        exit(0)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    trainlist_1 = open(os.path.join(args.dataset_1, 'trainlist.txt')).read().splitlines()
    validlist_1 = open(os.path.join(args.dataset_1, 'validlist.txt')).read().splitlines()
    print('Training set 1 size:', len(trainlist_1))
    print('Validation set 1 size:', len(validlist_1))

    trainlist_2 = open(os.path.join(args.dataset_2, 'trainlist.txt')).read().splitlines()
    validlist_2 = open(os.path.join(args.dataset_2, 'validlist.txt')).read().splitlines()
    print('Training set 1 size:', len(trainlist_2))
    print('Validation set 1 size:', len(validlist_2))

    index_1 = 0
    index_2 = 0
    step_1 = args.rate_1
    step_2 = args.rate_2

    train_1 = len(trainlist_1)
    test_1 = len(validlist_1)
    train_2 = len(trainlist_2)
    test_2 = len(validlist_2)

    trainlist = []
    validlist = []

    random.shuffle(trainlist_1)
    random.shuffle(validlist_1)
    random.shuffle(trainlist_2)
    random.shuffle(validlist_2)

    while True:
        trainlist.extend(trainlist_1[index_1:index_1+step_1])
        trainlist.extend(trainlist_2[index_2:index_2+step_2])
        index_1 += step_1
        index_2 += step_2
        if index_1 >= train_1 or index_2 >= train_2:
            break

    index_12 = 0
    index_22 = 0

    print('Training sets merged.')

    while True:
        validlist.extend(validlist_1[index_12:index_12+step_1])
        validlist.extend(validlist_2[index_22:index_22+step_2])
        index_12 += step_1
        index_22 += step_2
        if index_12 >= test_1 or index_22 >= test_2:
            break

    print('Testing sets merged.')

    print('Number of sequences:', len(trainlist), '(train),', len(validlist), '(test)')
    print('Entries from dataset 1:', index_1, '(train),', index_12, '(test)')
    print('Entries from dataset 2:', index_2, '(train),', index_22, '(test)')

    with open(os.path.join(args.output_dir, 'trainlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(trainlist))
        myfile.write('\n')
    with open(os.path.join(args.output_dir, 'validlist.txt'), mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(validlist))
        myfile.write('\n')

if __name__ == '__main__':
    mix()
    exit(0)