#!/bin/bash

git submodule update --init --recursive

mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=clang++ -DSHROUD_EXECUTABLE=/usr/local/bin/shroud ..
make -j 3 generate_umpire_shroud

