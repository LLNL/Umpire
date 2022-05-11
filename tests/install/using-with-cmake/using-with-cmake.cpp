//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

int main()
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");
  std::cout << "Got allocator: " << alloc.getName() << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAllocatorNames()) {
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  return 0;
}
