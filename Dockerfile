FROM axom/compilers:gcc-8 AS gcc
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 3 
RUN cd build && make test


FROM axom/compilers:clang-6 AS clang
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 3 
RUN cd build && make test


FROM axom/compilers:nvcc-9 AS nvcc
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On ..
RUN cd build && make -j 3 
