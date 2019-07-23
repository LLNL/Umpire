#!/usr/bin/env zsh
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

setopt extended_glob

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

LIC_CMD=$(which lic)
if [ ! $LIC_CMD ]; then
  echo "${RED} [!] This script requires the lic command."
  exit 255
fi

echo "Applying licenses to files"

files_no_license=$(grep -L 'This file is part of Umpire.' \
  benchmarks/**/*(^/) \
  cmake/**/*(^/) \
  docs/**/*~*rst(^/)\
  examples/**/*(^/) \
  host-configs/**/*(^/) \
  scripts/**/*(^/) \
  src/**/*~*tpl*(^/) \
  tests/**/*(^/) \
  CMakeLists.txt umpire-config.cmake.in)

echo $files_no_license | xargs $LIC_CMD -f scripts/umpire-license.txt 

echo "${GREEN} [Ok] License text applied. ${NOCOLOR}"
