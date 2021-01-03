#!/bin/bash

rm -rf Vimeo90k
mkdir Vimeo90k

./getVimeo90k3.sh
./getVimeo90k7.sh

echo Vimeo90k downloaded and extracted