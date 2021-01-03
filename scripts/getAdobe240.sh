#!/bin/bash

mkdir Adobe240
cd Adobe240

wget http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
unzip DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
rm DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
rm -r __MACOSX
mkdir original_high_fps_videos/videos

for f in original_high_fps_videos/*; do
	case "$f" in
	*.mov)
		dir=${f%.mov};;
	*.MOV)
		dir=${f%.MOV};;
	*.mp4)
		dir=${f%.mp4};;
	*.MP4)
		dir=${f%.MP4};;
	*.m4v)
		dir=${f%.m4v};;
	esac
	tbr=$(ffmpeg -i "$f" 2>&1 | sed -n "s/.*, \(.*\) tbr.*/\1/p")
	mkdir $dir
	ffmpeg -i "$f" -vf fps=$tbr $dir/out%d.png
	mv "$f" original_high_fps_videos/videos/
	echo $dir
done
