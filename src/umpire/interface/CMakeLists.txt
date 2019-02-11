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
add_shroud(
  YAML_INPUT_FILE umpire_shroud.yaml
  YAML_OUTPUT_DIR yaml
  C_FORTRAN_OUTPUT_DIR c_fortran
  PYTHON_OUTPUT_DIR python
)
