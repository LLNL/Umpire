#!/bin/bash

git config --global --add safe.directory /github/workspace
git submodule update --init --recursive

mkdir build && cd build 
cmake -DCMAKE_CXX_COMPILER=clang++ ..
make style

