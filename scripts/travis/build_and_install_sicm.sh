#!/usr/bin/env bash

set -e

# install dependencies
sudo apt-get install -y libnuma-dev

PREFIX="/tmp"

# install jemalloc
git clone --depth 1 https://github.com/jemalloc/jemalloc.git
cd jemalloc
mkdir build
./autogen.sh
cd build
../configure --prefix=${PREFIX}/jemalloc --with-jemalloc-prefix=je_
make
make install -i
cd ../..

# install SICM
git clone --depth 1 https://github.com/lanl/SICM.git
cd SICM
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX}/SICM -DJEMALLOC_INCLUDE_DIR=${PREFIX}/jemalloc/include -DJEMALLOC_LIBRARIES=${PREFIX}/jemalloc/lib/libjemalloc.so
make
make install
