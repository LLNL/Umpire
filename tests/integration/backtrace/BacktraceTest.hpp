//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_BACKTRACETEST_INCLUDE
#define UMPIRE_BACKTRACETEST_INCLUDE

class BacktraceTest {
public:
  BacktraceTest();
  void run();

private:
  void level1();
  void level2();
  void level3();
};
#endif // UMPIRE_BACKTRACETEST_INCLUDE
