#!/bin/bash

for f in $1/*; do
    if [ -d "$f" ]; then
        convert -delay 1 $f/*.png -loop 0 $f.gif
        rm -rf $f
        echo $f
    fi
done
