FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-7.3.0 AS gcc7
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-8.1.0 AS gcc8
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-9.4.0 AS gcc9
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-11.2.0 AS gcc
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-11.0.0 AS clang11
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-13.0.0 AS clang
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/umpire/workspace
RUN mkdir build && cd build && cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/cuda:cuda-10.1.243-ubuntu-18.04 AS nvcc10
ENV GTEST_COLOR=1
COPY . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc
ENV GTEST_COLOR=1
COPY . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11-debug
ENV GTEST_COLOR=1
COPY . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc
ENV GTEST_COLOR=1
COPY . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    mkdir build && cd build && \
    cmake -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM axom/compilers:rocm AS hip
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
ENV HCC_AMDGPU_TARGET=gfx900
RUN mkdir build && cd build && cmake -DROCM_ROOT_DIR=/opt/rocm/include -DHIP_RUNTIME_INCLUDE_DIRS="/opt/rocm/include;/opt/rocm/hip/include" -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DENABLE_HIP=On ..
RUN cd build && make -j 16

FROM ghcr.io/rse-ops/intel-ubuntu-20.04:intel-2021.2.0 AS sycl
ENV GTEST_COLOR=1
COPY . /home/umpire/workspace
WORKDIR /home/axom/workspace
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=dpcpp -DENABLE_WARNINGS_AS_ERRORS=Off -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On -DUMPIRE_ENABLE_SYCL=On .."
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh && cd build && make -j 16"
