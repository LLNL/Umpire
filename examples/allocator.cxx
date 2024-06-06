//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <iostream>

int main() {
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator("HOST");

  std::cout << "Got allocator: " << alloc.getName() << std::endl;

  std::cout << "Available allocators: ";
  for (auto s : rm.getAllocatorNames()){
    std::cout << s << "  ";
  }
  std::cout << std::endl;

  return 0;
}
