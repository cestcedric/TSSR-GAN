# TSSR-GAN

## Table of Contents

1. [Setup](#Setup)
1. [Interpolation](#Interpolation)
1. [Training](#Training)
1. [Notes](#Notes)
1. [TODO](#TODO)

----

This respository contains code and material for the TSSR-GAN project, a GAN for Temporal and Spatial Super-Resolution.
Shoutout to the authors of [DAIN](https://github.com/baowenbo/DAIN) and [TecoGAN](https://github.com/thunil/TecoGAN).
TSSR-GAN uses a lot of modules of these two networks.

Included in this repository:

- Setup guide
- Scripts for data download and preparation
- Scripts to train TSSR-GAN
- Scripts for inference and benchmarking
- Google Colab Notebook

Everything is implemented in Python 3 and tested for PyTorch 1.4 with CUDA 10.0 or CUDA 10.1, but should also run with PyTorch >= 1.4, it is just not tested.

---
## Setup
1. Clone git `git clone https://github.com/cestcedric/TSSR-GAN.git`
2. Create python environment and install packages listed in `requirements.txt`. These packages also contain packages for the metric computation and probably a few not necessary anymore. PyTorch version just has to be >= 1.4, but you have to match CUDA from PyTorch with the NVCC version.
3. Compile PWCNet and DAIN modules (make sure that your PyTorch CUDA version and NVCC version match): `scripts/setup.sh`
4. Download model weights: `scripts/getModelWeights.sh`

---
## Inference

Add some sequence base frames and put them into this structure:
```
main_dir
    sequence_1
        start.png, end.png
    sequence_2
        start.png, end.png
```

Then start the super-resolution with:
```
python interpolate.py \
--input_dir /path/to/main_dir/ \
--output_dir /path/to/results \
--weights model_weights/pretrained/best_GEN.pth \
--timestep 0.5
```

A time step of 0.5 means one intermediate frame, so start <-0.5-> intermediate <-0.5-> end. For 9 intermediate frames use time step = 1/(intermediate frames + 1) = 0.1.

---


## Training

Don't train on Google Colab, unless you have a paid version or something, it takes way to long to get results before the automatic disconnection using the base version.

Anyway, loads of parameters that can be set for training, see `utils/my_args.py` or use `!python3 train.py --help`.

The training data can be saved randomly dispersed, all in one directory, whatever. BUT you have to provide a directory with `trainlist.txt` and `validlist.txt`, which look something like this for sequences with 3 frames:
```
seq1/frame1.png;seq1/frame2.png;seq1/frame3.png
seq2/frame1.png;seq2/frame2.png;seq2/frame3.png
seq3/frame1.png;seq3/frame2.png;seq3/frame3.png
seq4/frame1.png;seq4/frame2.png;seq4/frame3.png
seq5/frame1.png;seq5/frame2.png;seq5/frame3.png
```

Start training using:
```
! python3 train.py \
--pretrained_generator path/to/pretrained.pth \
--numEpoch 20 \
--lengthEpoch 5000 \
--datasetPath path/to/dataset \
--max_img_height 512 \
--max_img_width 512 \
--interpolationsteps 5 \
--improved_estimation 1 \
--gan \
--depth \
--flow \
--spatial \
--temporal \
--loss_pingpong \
--loss_layer \
--loss_perceptual \
--debug_output \
--tb_experiment training \
--debug_output_dir training_output/training \
--warmup_discriminator \
--warmup_boost 4 \
--no_cudnn_benchmark
```

Interpolationsteps denotes the newly generated intermediate frames, which means interpolationsteps = sequence length - 2.

---

# ENJOY!