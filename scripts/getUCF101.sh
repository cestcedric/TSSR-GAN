#!/bin/bash

mkdir UCF
cd UCF

mkdir UCF101
cd UCF101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar e UCF101.rar
rm UCF101.rar
cd ..

# remove additional clips, we just need one per scene for test purposes
# obviously remove that part if you want to convert all clips, e.g. for training
rm UCF101/*g1*
rm UCF101/*g2*
rm UCF101/*g02*
rm UCF101/*g03*
rm UCF101/*g04*
rm UCF101/*g05*
rm UCF101/*g06*
rm UCF101/*g07*
rm UCF101/*g08*
rm UCF101/*g09*
rm UCF101/*c02*
rm UCF101/*c03*
rm UCF101/*c04*
rm UCF101/*c05*
rm UCF101/*c06*
rm UCF101/*c07*

for f in UCF101/*; do
	dir=${f%.avi}
	mkdir $dir
	convert $f "$dir%d.png"
	mv $dir*.* $dir/
	echo $dir
done
