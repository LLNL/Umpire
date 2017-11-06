#!/bin/bash

if [[ $HOSTNAME == *manta* ]]; then
  bsub -x -n 1 -G guests -Ip ctest -T Test
else
  srun -ppdebug -t 5 -N 1 ctest -T Test
fi
