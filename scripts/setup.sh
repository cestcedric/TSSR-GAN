cd my_package
chmod +x build.sh
./build.sh
cd ../PWCNet/correlation_package_pytorch1_0
chmod +x build.sh
./build.sh
cd ..
cd ..
rm -rf model_weights
rm -rf MiddleBurySet
mkdir model_weights
mkdir MiddleBurySet
cd model_weights
wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth
cd ../MiddleBurySet
wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
unzip other-color-allframes.zip
wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
unzip other-gt-interp.zip
cd ..