FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-7.3.0 AS gcc7
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-8.1.0 AS gcc8
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-18.04:gcc-9.4.0 AS gcc9
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-18.04:gcc-11.2.0 AS gcc11
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-10.0.0 AS clang10
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-11.0.0 AS clang11
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-12.0.0 AS clang12
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-13.0.0 AS clang13
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/cuda:cuda-10.1.243-ubuntu-18.04 AS nvcc10
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM axom/compilers:rocm-4.3.1 AS hip
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN cmake -DENABLE_WARNINGS_AS_ERRORS=Off -DCMAKE_CXX_COMPILER=/opt/rocm-4.3.1/llvm/bin/amdclang++ -DHIP_PATH=/opt/rocm-4.3.1/hip -DROCM_PATH=/opt/rocm-4.3.1 -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DENABLE_HIP=On .. && \
    make VERBOSE=1

FROM axom/compilers:rocm-4.3.1 AS hip.debug
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build

FROM ghcr.io/rse-ops/intel-ubuntu-20.04:intel-2021.2.0 AS sycl
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace/build
RUN /bin/bash -c "source /opt/view/setvars.sh && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DENABLE_WARNINGS_AS_ERRORS=Off -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DUMPIRE_ENABLE_SYCL=On .. && \
    make -j 16"

