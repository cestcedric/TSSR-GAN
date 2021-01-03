import argparse
from   datetime import datetime
import os
from   PIL import Image


def rescale(dir_in, dir_out, scale):
    for i in os.listdir(dir_in):
        img = Image.open(os.path.join(dir_in, i))
        img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.LANCZOS)
        img.save(os.path.join(dir_out, i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training sequences')
    parser.add_argument('--input_path', type=str, default='input', help='output directory (default: ./input)')
    parser.add_argument('--output_path', type=str, default='output', help='output directory (default: ./output)')
    parser.add_argument('--scale', type=float, default=0.25, help= 'Scaling factor (default: 0.25)')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    dirs = [f.path for f in os.scandir(args.input_path) if f.is_dir()]
    for i, dir in enumerate(dirs):
        d = dir.split('/')[-1]
        out = os.path.join(args.output_path, d)
        if not os.path.exists(out):
            os.mkdir(out)
        rescale(dir_in = dir, dir_out = out, scale = args.scale)
        print('Rescaled', i+1, 'of', len(dirs))