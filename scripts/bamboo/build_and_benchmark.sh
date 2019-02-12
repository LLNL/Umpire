#!/bin/bash
##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
##############################################################################

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

. ${SCRIPTPATH}/build_and_test.sh

echo "Benchmarking..."
COMMIT=`git rev-parse --short HEAD`
DATE=`date +%Y-%m-%d`
BENCHMARK_OUTPUT_NAME="${COMMIT}_${COMPILER}_${SYS_TYPE}_${DATE}"

if [[ $HOSTNAME == *manta* ]]; then
  bsub -x -n 1 -G guests -Ip ./benchmark/copy_benchmarks --benchmark_out=copy_$BENCHMARK_OUTPUT_NAME.json --benchmark_out_format=json
  bsub -x -n 1 -G guests -Ip ./benchmark/allocator_benchmarks --benchmark_out=allocator_$BENCHMARK_OUTPUT_NAME --benchmark_out_format=json
else
  srun -ppdebug -t 5 -N 1 ./benchmark/copy_benchmarks --benchmark_out=copy_${BENCHMARK_OUTPUT_NAME}.json --benchmark_out_format=json
  srun -ppdebug -t 5 -N 1 ./benchmark/allocator_benchmarks --benchmark_out=allocator_${BENCHMARK_OUTPUT_NAME}.json --benchmark_out_format=json
fi

cp copy_${BENCHMARK_OUTPUT_NAME}.json /usr/workspace/wsrzc/umpire/benchmark_results
cp allocator_${BENCHMARK_OUTPUT_NAME}.json /usr/workspace/wsrzc/umpire/benchmark_results
