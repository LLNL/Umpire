FROM ghcr.io/llnl/radiuss:ubuntu-22.04-gcc-13 AS gcc
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:clang-15-ubuntu-22.04 AS clang
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure


FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-11.0.0 AS umap_build
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN \
    git clone -q https://github.com/LLNL/umap.git umap && \
    cd umap && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/umpire/install.umap .. && \
    make -j 16 install && cd ../.. && \
    cmake -DUMAP_ROOT=/home/umpire/install.umap -DCMAKE_INSTALL_PREFIX=/home/umpire/install.umpire -DUMPIRE_ENABLE_UMAP=On -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16

FROM ghcr.io/llnl/radiuss:clang-15-ubuntu-22.04 AS asan
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
          -DUMPIRE_ENABLE_C=On -DCMAKE_CXX_FLAGS="-fsanitize=address" -DENABLE_TESTS=On -DUMPIRE_ENABLE_TOOLS=On \
          -DUMPIRE_ENABLE_ASAN=On -DUMPIRE_ENABLE_SANITIZER_TESTS=On .. && \
    make -j 2 && \
    ctest -T test -E operation_tests --output-on-failure

FROM ghcr.io/llnl/radiuss:ubuntu-22.04-cuda-12-3 AS cuda
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=70 .. && \
    make -j 16

# TODO: switch to ROCM 6
FROM ghcr.io/llnl/radiuss:hip-5.6.1-ubuntu-20.04 AS hip
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DENABLE_WARNINGS_AS_ERRORS=Off -DCMAKE_CXX_COMPILER=/opt/rocm-5.6.1/bin/amdclang++ -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DENABLE_HIP=On .. && \
    make -j 16 VERBOSE=1

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS sycl
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_C_COMPILER=icx -DENABLE_WARNINGS_AS_ERRORS=Off -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DUMPIRE_ENABLE_SYCL=On .. && \
    make -j 16"

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_WARNINGS_AS_ERRORS=Off -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On .. && \
    make -j 16"

