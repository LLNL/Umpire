#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

. ${SCRIPTPATH}/build_and_test.sh

#
# The remainder of this script assumes that build_and_test.sh places us in the
# build directory (this assumption is not new, I am just documenting it now).
#
echo "Benchmarking..."
COMMIT="$( cd "$(dirname "$0")" ; git rev-parse --short HEAD )"
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
