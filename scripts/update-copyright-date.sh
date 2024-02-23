#!/bin/bash
##############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

cd `git rev-parse --show-toplevel`
for f in `git ls-tree -r develop --name-only`
do
  if [ -f $f ]
  then
    if grep -q 2016-24 $f
    then
      echo "Updating $f"
      sed -i.bak -e 's/2016-24/2016-24/g' $f
      rm $f.bak
    fi
  fi
done
