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
  auto alloc = rm.getAllocator("NO_OP");

  char* my_array = static_cast<char*>(alloc.allocate(100 * sizeof(char)));

  alloc.deallocate(my_array);
}
