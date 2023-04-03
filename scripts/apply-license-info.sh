#!/usr/bin/env zsh
##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

setopt extended_glob

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

LIC_CMD=`which lic`
if [ ! $LIC_CMD ]; then
  echo "${RED} [!] This script requires the lic command."
  exit 255
fi

echo "Applying licenses to files"

for d in benchmarks cmake docs examples host-configs scripts src tests CMakeLists.txt umpire-config.cmake.in; do
  for x in `grep -lr 'SPDX-License-Identifier' $d --exclude-dir=tpl --exclude-dir=blt --exclude=.gitmodules --exclude=.gitignore`; do
    if [ $x != scripts/umpire-license.txt ]; then
      $LIC_CMD -f scripts/umpire-license.txt $x
    fi
  done
done

#
# Remove the files with "~" appended to them
#
git clean -fd -q

echo "${GREEN} [Ok] License text applied. ${NOCOLOR}"
