//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "BacktraceTest.hpp"
#include "umpire/util/Macros.hpp"

int main(int UMPIRE_UNUSED_ARG(argc), char** UMPIRE_UNUSED_ARG(argv)) {
  BacktraceTest test;

  test.run();

  return 0;
}
