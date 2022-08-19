#!/bin/bash
cmake -DENABLE_MPI=On -DENABLE_HIP=On -DCMAKE_CXX_COMPILER=/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.16-rocmcc-5.2.0/bin/mpicxx -DCMAKE_C_COMPILER=/usr/tce/packages/cray-mpich-tce/cray-mpich-8.1.16-rocmcc-5.2.0/bin/mpicc -DCMAKE_HIP_ARCHITECTURES=gfx90a -DROCM_PATH=/opt/rocm-5.2.0 ../
srun -n 1 make -j
