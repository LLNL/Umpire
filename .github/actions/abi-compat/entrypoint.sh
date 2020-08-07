#!/bin/bash

git submodule update --init --recursive

mkdir build && cd build 
cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_FLAGS="-Og" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ -DBUILD_SHARED_LIBS=On ..
make -j 3 umpire
cd ..

git checkout origin/main
git submodule update --init --recursive

mkdir build-main && cd build-main 
cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_FLAGS="-Og" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ -DBUILD_SHARED_LIBS=On ..
make -j 3 umpire
cd ..

abi-dumper build-main/lib/libumpire.so -o umpire-main.dump -lver 3.0.0
abi-dumper build/lib/libumpire.so -o umpire-dev.dump -lver 4.0.0

abi-compliance-checker -l umpire -old umpire-main.dump -new umpire-dev.dump
