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

- Setup guide, including requirements.txt and precompiled binaries that need specific nvcc versions
- Scripts for data download and preparation
- Scripts to train TSSR-GAN and a subnetwork
- Scripts for interpolation and benchmarking

TODO: some sample images or cool GIFs like for TecoGAN

### Setup

TODO: update `setup.sh` and `setup_precompiled.sh` to do all compilation/installation

Remember to make shell scripts executable with `chmod +x script.sh` before trying to execute them.

1. Clone git: `git clone TODO:`
2. Create python environment and install packages listed in `requirements.txt`
3. Compile DAIN modules with cuda 9.0: `.\setup.sh` or install precompiled packages `.\setup_precompiled`
4. Download model weights: `.\getModelWeights.sh` TODO: upload somewhere
5. Download sample interpolation data: TODO: getMiddlebury script

### Interpolation

TODO: sample interpolation on Middlebury script

### Training

1. Download training data: `.\getVimeo90k`, `.\getNeedForSpeed.sh`
2. Prepare data set files: TODO: make shell script to call python scripts with parameters
3. Start training: `TODO: turn best slurm job into .sh file for easy training command`

----

## Notes

### Ablation study

| SLURM ID | Model ID | Dataset | Experiment | Runtime | Pretrained | Description | Notes | Memory Usage |
|----------|----------|---------|------------|---------|------------|-------------|-------|--------------|
| 284827 | ablation_spatial | Mix50 | ablation | 4d 12h | best_upscaling_warm.pth | - | - | UsedVRAM0 maximum 45 %, TotalVRAM0 24449 MB |
| 284828 | ablation_spatiotemporal | Mix50 | ablation | 5d 2h | best_upscaling_warm.pth | - | - | UsedVRAM0 maximum 48 %, TotalVRAM0 22916 MB |
| 289704 | ablation_spatiotemporal_2 | Mix50 | ablation | - | best_upscaling_warm.pth | - | - | - |
| 284829 | ablation_perceptual | Mix50 | ablation | 5d | best_upscaling_warm.pth | - | - | UsedVRAM0 maximum 48 %, TotalVRAM0 22916 MB |
| 284830 | ablation_pingpong | Mix50 | ablation | 7d 1h | best_upscaling_warm.pth | - | - | UsedVRAM0 maximum 96 %, TotalVRAM0 22916 MB |
| 284831 | ablation_lowresolution | Mix50 | ablation | 6d 21h | best_upscaling_warm.pth | - | - | UsedVRAM0 maximum 98 %, TotalVRAM0 22916 MB |
| 288186 | TSSR_1 | Mix50 | training | - | best_upscaling_warm.pth | - | - | - |
| 288187 | TSSR_2 | Mix50 | training | +2d | best_upscaling_warm.pth | - | - | - |
| 288188 | TSSR_3 | Mix50 | training | +2d | best_upscaling_warm.pth | - | - | - |
| 289705 | TSSR_4 | Mix50 | training | - | best_upscaling_warm.pth | - | - | - |

## TODO

### Ablation

- TODO: Improved estimation: With, Without

### Benchmarking

- TODO: Triplets: ZSM, DAIN + Bicubic, DAIN + TecoGAN, Overlay + Bicubic, TSSR
- TODO: Septuplets: DAIN + Bicubic, DAIN + TecoGAN, Overlay + Bicubic, TSSR

### 2.3 Miscellaneous

- TODO: remove or generalize testset_adb.py, rename_adb.py and mdb
- TODO: standardize indentation
- TODO: filter requirements.txt to only include necessary packages
