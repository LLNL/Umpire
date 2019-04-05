#!/bin/sh

git submodule update --init --recursive

mkdir build && cd build

cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DENABLE_CUDA=Off -DCMAKE_EXPORT_COMPILE_COMMANDS=On ..

/run-clang-tidy.py -j 8 -header-filter='.*' -checks='*' -fix
