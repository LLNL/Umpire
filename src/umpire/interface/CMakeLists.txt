##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
add_shroud(
  YAML_INPUT_FILE umpire_shroud.yaml
  YAML_OUTPUT_DIR yaml
  C_FORTRAN_OUTPUT_DIR c_fortran
  PYTHON_OUTPUT_DIR python
)
