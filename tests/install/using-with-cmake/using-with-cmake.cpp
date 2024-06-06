//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "fmt/format.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

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
