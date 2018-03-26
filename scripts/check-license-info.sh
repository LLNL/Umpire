#!/usr/bin/env zsh

# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

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

if [ $files_no_license ]; then
  print "${RED} [!] Some files are missing license text: ${NOCOLOR}"
  echo "${files_no_license}"
  exit 255
else
  print "${GREEN} [Ok] All files have required license info."
  exit 0
fi
