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
replay_tests_dir=$1
tools_dir=$2
testprogram=$replay_tests_dir/replay_tests
replayprogram=$tools_dir/replay

#
# The following program will generate a CSV file of Umpire activity that
# may be replayed.
#
UMPIRE_REPLAY="On" $testprogram
if [ $? -ne 0 ]; then
    echo "Failed: Unable to run $testprogram"
    exit 1
fi

#
# Now replay from the activity captured in the replay_test1.csv file
#
$replayprogram -i umpire.0.*.0.replay -t replay.out
if [ $? -ne 0 ]; then
    echo "Failed: Unable to run $replayprogram"
    exit 1
fi

#
# Now, compare the results being careful to allow for different object
# references (everything else should be the same).
#
diff replay.out $replay_tests_dir/test_output.good
if [ $? -ne 0 ]; then
    echo "Diff failed"
    exit 1
fi

/bin/rm -f replay.out umpire*replay umpire*log
exit 0
