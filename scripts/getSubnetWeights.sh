#!/bin/bash

cd submodules/MegaDepth
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir test_local
cd test_local
wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best_generalization_net_G.pth
cd ../../../PWCNet
wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/pwc_net.pth.tar
