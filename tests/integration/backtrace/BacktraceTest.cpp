//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "BacktraceTest.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

BacktraceTest::BacktraceTest()
{
}

void BacktraceTest::run()
{
  level1();
}

void BacktraceTest::level1()
{
  level2();
}

void BacktraceTest::level2()
{
  level3();
}

void BacktraceTest::level3()
{
  auto& rm = umpire::ResourceManager::getInstance();
  try {
  auto alloc = rm.getAllocator("NoneExistantAllocatorName");
  UMPIRE_USE_VAR(alloc);
  } catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}
