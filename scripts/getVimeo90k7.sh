#!/bin/bash

rm -rf Vimeo90k/Vimeo90k7
mkdir Vimeo90k/Vimeo90k7

cd Vimeo90k
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip vimeo_septuplet.zip
rm vimeo_septuplet.zip

cd ../..

echo Vimeo90k7 downloaded and extracted