#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

containing_dir=`dirname "${BASH_SOURCE[0]}"`
if ! cd ${containing_dir}
then
  "Failed to go to containing test directory ${containing_dir}"
  exit 2
fi

root_dir=`git rev-parse --show-toplevel`
if ! cd ${root_dir}
then
  "Failed to go to source root directory ${root_dir}"
  exit 3
fi

#
# This script uses grep, awk, and sed to accomplish the following:
# 1) Find all UMPIRE_REPLAY calls in the umpire source code (uses 'git' to
#    determine root directory of workspace.
# 2) From the collected UMPIRE_REPLAY events, check to see if that event is
#    also in the replay interpreter implementation.
#
# This script will exit with code 1 in the failure case and will display the
# names of the events it could not find.
#
# Otherwise, exit-code is used to say that all events are handled in the replay
# tool's implementation.
#

ecode=0
for event in $(\
  grep -rl -I --exclude-dir "./tools/replay" --exclude-dir "./tests/integration/replay" --exclude-dir "./docs" --exclude ./src/umpire/Replay.hpp UMPIRE_REPLAY . |\
  xargs cat |\
  awk '/UMPIRE_REPLAY/,/;/ {print}' |\
  sed -e 's/\\//g' -e 's/\"//g' -e 's/ *//g' -e 's/UMPIRE_REPLAY(//g' | grep "event" | sed -e 's/.*event://g' -e 's/,.*//g' -e 's/).*//' | sort | uniq)
do
  x='"'$event'"'
  if ! grep -q $x ./tools/replay/ReplayInterpreter.cpp
  then
    ecode=1
    echo "The replay program cannot handle the $x replay event"
  fi
done
exit $ecode
