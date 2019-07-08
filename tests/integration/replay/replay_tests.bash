#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
replay_tests_dir=$1
tools_dir=$2
testprogram=$replay_tests_dir/replay_tests
diffprogram=$tools_dir/replaydiff

#
# The following program will generate a CSV file of Umpire activity that
# may be replayed.
#
echo UMPIRE_REPLAY="On" $testprogram
UMPIRE_REPLAY="On" $testprogram
if [ $? -ne 0 ]; then
    echo "Failed: Unable to run $testprogram"
    exit 1
fi

echo $diffprogram $replay_tests_dir/test_output.good umpire.0.*.0.replay
$diffprogram $replay_tests_dir/test_output.good umpire.0.*.0.replay
if [ $? -ne 0 ]; then
    echo "Diff failed"
    /bin/rm -f replay.out umpire*replay umpire*log
    exit 1
fi

pwd
/bin/rm -f replay.out umpire*replay umpire*log
exit 0
