#!/bin/bash
##############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

cd `git rev-parse --show-toplevel`
for f in `git ls-tree -r develop --name-only`
do
  if [ -f $f ]
  then
    if grep -q 2016-20 $f
    then
      # grep 2016-20 $f
      echo "Updating $f"
      sed -i -e 's/2016-20/2016-21/g' $f
    fi
  fi
  # for i in `grep 2016-19 * -R -l`; do sed -i.bak 's/2016-19/2016-20/g' $i; done
done
