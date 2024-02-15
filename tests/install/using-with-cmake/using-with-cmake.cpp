//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "fmt/format.h"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  std::cout << fmt::format("Got allocator: {0}", alloc.getName()) << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAllocatorNames()) {
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  return 0;
}
