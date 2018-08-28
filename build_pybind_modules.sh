#!/bin/sh
# This is a comment!

cd src/math_utils/
mkdir build
cd build
cmake ..
make
cd ../../pcl_utils/
mkdir build
cd build
cmake ..
make
cd ../../../
