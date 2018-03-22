###########################################################
# Example host-config file for the quartz cluster at LLNL
###########################################################
#
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers
# 
###########################################################

###########################################################
# pgi@17.10 compilers
###########################################################

# c compiler
set(CMAKE_C_COMPILER "/usr/tce/packages/pgi/pgi-17.10/bin/pgcc" CACHE PATH "")

# cpp compiler
set(CMAKE_CXX_COMPILER "/usr/tce/packages/pgi/pgi-17.10/bin/pgc++" CACHE PATH "")

# fortran support
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/pgi/pgi-17.10/bin/pgfortran" CACHE PATH "")

