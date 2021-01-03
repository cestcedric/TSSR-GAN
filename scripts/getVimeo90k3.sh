#!/bin/bash

rm -rf Vimeo90k/Vimeo90k3
mkdir Vimeo90k/Vimeo90k3

cd Vimeo90k
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip vimeo_triplet.zip
rm vimeo_triplet.zip

cd ../..

echo Vimeo90k3 downloaded and extracted