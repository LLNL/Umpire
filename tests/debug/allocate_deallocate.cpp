//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include <iostream>

int main() {
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.getAllocator("HOST");

  char* my_array = static_cast<char*>(alloc.allocate(1<<26));

  alloc.deallocate(my_array);
}
