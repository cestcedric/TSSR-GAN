#!/bin/bash

unzip precompiled.zip
cp -r precompiled/dist-packages $VIRTUAL_ENV/lib/python3.6/
rm -rf precompiled

echo Successfully installed precompiled packages
