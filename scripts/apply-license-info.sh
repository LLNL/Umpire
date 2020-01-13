#!/usr/bin/env zsh
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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
    $LIC_CMD -f ~/umpire-license.txt $x
  done
done

git clean -fd -q

echo "${GREEN} [Ok] License text applied. ${NOCOLOR}"
