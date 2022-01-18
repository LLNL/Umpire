#!/bin/bash

git submodule update --init --recursive

mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=clang++ ..
make style

