//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/ResourceManager.hpp"
#include "umpire/util/Macros.hpp"

int main(int UMPIRE_UNUSED_ARG(argc), char** UMPIRE_UNUSED_ARG(argv))
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto alloc = rm.getAllocator("NoneExistantAllocatorName");
    UMPIRE_USE_VAR(alloc);
  } catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
