#!/bin/bash



echo "Benchmarking..."
if [[ $HOSTNAME == *manta* ]]; then
  bsub -x -n 1 -G guests -Ip ./benchmark/copy_benchmarks --benchmark_out=copy_$1.json --benchmark_out_format=json
  bsub -x -n 1 -G guests -Ip ./benchmark/allocator_benchmarks --benchmark_out=allocator_$1 --benchmark_out_format=json
else
  srun -ppdebug -t 5 -N 1 ./benchmark/copy_benchmarks --benchmark_out=copy_${1}.json --benchmark_out_format=json
  srun -ppdebug -t 5 -N 1 ./benchmark/allocator_benchmarks --benchmark_out=allocator_${1}.json --benchmark_out_format=json
fi
