#!/bin/bash

# create directory, switch into it
mkdir NeedForSpeed
cd NeedForSpeed

# initial Download
curl -fSsl http://ci2cv.net/nfs/Get_NFS.sh | bash -

# extract
# nohup if you want to close terminal and let it run
unzip NeedForSpeed/\*.zip

# sort data into 240 FPS and 30 FPS folders
mkdir NeedForSpeed/30
mkdir NeedForSpeed/240
mv NeedForSpeed/*/30/* 30
mv NeedForSpeed/*/240/* 240

# remove original folders
rm -rf NeedForSpeed/[a-z]*

# remove annotation files
rm NeedForSpeed/30/*.txt
rm NeedForSpeed/240/*.txt

# regroup files, create train/test set
cd ..
python utils/createSequences.py