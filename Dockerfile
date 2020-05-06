FROM axom/compilers:gcc-5 AS gcc5
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:gcc-6 AS gcc6
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:gcc-7 AS gcc7
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:gcc-8 AS gcc
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:clang-4 AS clang4
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:clang-5 AS clang5
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:clang-6 AS clang
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && make test

FROM axom/compilers:nvcc-9 AS nvcc
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On ..
RUN cd build && make -j 16

FROM axom/compilers:rocm AS hip
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
ENV HCC_AMDGPU_TARGET=gfx900
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DENABLE_HIP=On ..
RUN cd build && make VERBOSE=1

FROM axom/compilers:oneapi AS sycl
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
ENV HCC_AMDGPU_TARGET=gfx900
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=dpcpp -DENABLE_DEVELOPER_DEFAULTS=On -DENABLE_SYCL=On ..
RUN cd build && make VERBOSE=1